#!/usr/bin/env python
# coding: utf-8

# ### 1.0 引入依赖，准备文件 

# 我们先配置运行环境，并引入需要用到的势函数

# In[1]:


#! pip install matplotlib pymbar optax
#! conda install mdtraj -y
#! if [ ! -e DMFF ];then git clone https://gitee.com/deepmodeling/DMFF.git;fi
#! git config --global --add safe.directory `pwd`/DMFF
#! cd DMFF && git checkout devel
#! export XLA_PYTHON_CLIENT_PREALLOCATE=FALSE
#! cd DMFF && python setup.py install


# 除去 DMFF，我们还需要使用的包有 JAX、OpenMM；
# 
# - OpenMM：管理核心的力场文件和参数数据（力场参数读取的前端）
# - JAX：可微分框架（力场计算的后端引擎）
# 
# 同时，在后面的案例中我们还会用到一些其他库以及轨迹分析软件如mdtraj等，这里一并引入

# In[2]:


import sys
import numpy as np
import jax
import jax_md
import jax.numpy as jnp
from jax import value_and_grad, jit, vmap
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import dmff
from dmff import Hamiltonian, NeighborList
from dmff.api import Hamiltonian
from dmff.common import nblist
from dmff.optimize import MultiTransform, genOptimizer
from dmff.mbar import MBAREstimator, SampleState, TargetState, Sample, OpenMMSampleState, buildTrajEnergyFunction
import pickle
from pprint import pprint
import optax
import mdtraj as md
from itertools import combinations
import matplotlib.pyplot as plt


# 在这里，我们先通过脚本将GROMACS的拓扑文件转换成openmm的xml文件,DMFF只支持PeriodicTorsionForce类型的二面角，不支持RBTorsionForce。故需要把RBTorsionForce转换成PeriodicTorsionForce类型的二面角，但能量相差一个[常数项](http://docs.openmm.org/latest/userguide/theory/02_standard_forces.html)，对力没有影响

# In[3]:





# 接下来使用DMFF的引擎计算能量和受力。
# ### 1.1 读入现有力场参数和拓扑 | OpenMM前端 

# DMFF中有和 OpenMM 的 `ForceField` 类似的，读取力场参数的功能类 `Hamiltonian`，可以定义更广义的体系势能函数，同时又兼容对现有力场参数的读取：
# 
# - 我们可以使用OpenMM主要读入PDB和拓扑：
#   - 拓扑（`1.xml`）
#   - 坐标PDB（`1.pdb`）
# - 使用DMFF的Hamiltonian读入力场参数，以便建立可微分势能函数
#   - 力场文件 （`1.xml`）
# 
# DMFF势函数，除Hamiltonian名称外，和 OpenMM 的用法是相同的，OpenMM 力场的 XML 文件也可直接复用。

# In[5]:


app.Topology.loadBondDefinitions("1.xml")
pdb = app.PDBFile("1.pdb")
ff = Hamiltonian("1.xml")
potentials = ff.createPotential(pdb.topology)


# 在DMFF中，势函数参数和计算将会由JAX管理，例如上述DMFF势函数包括了在DMFF中重新实现的 HarmonicBondForce、HarmonicAngleForce、PeriodicTorsionForce、NonbondedForce，

# In[6]:


for k in potentials.dmff_potentials.keys():
    pot = potentials.dmff_potentials[k]
    print(k, pot)

params = ff.getParameters()


# ### 1.2 计算 | JAX可微分后端

# 上述定义的势函数的计算中，我们需要这样几个参数：
# 
# - **坐标** positions: pdb中坐标的精度只有0.0001 nm，故我们使用MDAnalysis读取原始结构
# 
# - **体系模拟的盒子定义** box: 我们的PDB文件中没有定义 box，所以需要加上 box 大小的定义；（当然，使用None也会得到结果，因为我们的体系并不涉及周期边界）
# 
# - **原子近邻表定义** pairs: 势函数计算能量的接口同时也需要输入Neighborlist以便计算nonbondforce，故也可以使用NeighborList类来得到pairs
# 
# 然后就可以传递给 potentials.dmff_potentials 中保存的，由`generator`解析XML中的力场参数生成的`get_energy`函数来计算相应的能量，例如可以计算`NonbondedForce`能量：

# In[7]:


import MDAnalysis
u = MDAnalysis.Universe("gromacs/hexane.gro")
positions = jnp.array(u.coord.positions/10)


# In[8]:


box = jnp.array([
    [100.0,  0.0,  0.0],
    [ 0.0, 100.0,  0.0],
    [ 0.0,  0.0, 100.0]
])
#box=None # 使用这个也可以,但好像没有办法生成pairs


# 由于LJ参数的混合规则，OPLS-AA力场必须要自定义[LennardJonesForce](http://docs.openmm.org/latest/userguide/application/05_creating_ffs.html)，但DMFF支持并不好，因此使用NonbondedJaxGenerator生成共价矩阵

# In[9]:


Generators_name = [i.name for i in ff.getGenerators()]


# In[10]:


NonbondedForce_index = Generators_name.index('NonbondedForce')


# In[11]:


nbList = NeighborList(box, r_cutoff=4, covalent_map=ff.getGenerators()[NonbondedForce_index].covalent_map)


# In[12]:


nbList.allocate(positions)


# In[13]:


pairs = nbList.pairs


# 计算二面角键作用项

# In[14]:


bdfunc = potentials.dmff_potentials['PeriodicTorsionForce']


# 可以用 inspect 看看，`bdfunc`是一个【函数】,而inspect.signature()方法会告诉我们这个函数的输入参数有哪些

# In[15]:


import inspect
print(inspect.signature(bdfunc))


# In[16]:


bd_ene = bdfunc(positions, box, pairs, params)
print(bd_ene,"kJ/mol")


# 或是计算体系总能量，对于我们上面定义的体系，其能量为：
# $$
# E_{\rm{total}}^{\rm{OPLS-AA}}=E_{\rm{bond}}+E_{\rm{angle}}+E_{\rm{torsion}}+E_{\rm{nonbond}}
# $$
# 

# 使用之前的 potential，调用`getPotentialFunc()`方法，即可获得计算总能量的函数。

# In[17]:


efunc = potentials.getPotentialFunc()
params = ff.getParameters()

tot_ene = efunc(positions, box, pairs, params)
print(tot_ene)


# JAX作为计算后端的最大优势，则在于我们可以使用`jax.grad`函数来获得函数的导函数，其语法为 `jax.grad(func, argnums)`，含义为对函数的第argnums参数求（偏）导。  
# 我们拿到的总能量计算函数`efunc`的接口是 [坐标、box、成键对、力场参数] （DMFF的经典力场实现是这组参数，**但不同Force可能有不同定义**）

# 我们“对函数求（偏）导函数”的操作，求总能量对坐标的偏导数，即可用于计算分子中的原子受力
# $$
# \frac{\partial{E_{\rm{total}}}}{\partial{\mathbf{Z_i}}}=-\mathbf{F_i}, \ i=x,y,z
# $$
# （注意力是能量导数反方向，所以需要取负）

# In[18]:


pos_grad_func = jax.grad(efunc, argnums=0)
force = -pos_grad_func(positions, box, pairs, params)
print(force)


# In[19]:




# 可以看到，OpenMM的计算值与DMFF的计算值非常接近。

# 接下来计算Hessian矩阵，即能量对原子坐标的二阶导数
# $$
# \mathbf{H}_{ij}=\frac{\partial ^2f}{\partial x_i\partial x_j}
# $$

# In[20]:

def calculate_cm(positions, box, pairs, params):
    pos_hessian_func = jax.hessian(efunc, argnums=0)
    hessian = pos_hessian_func(positions, box, pairs, params)


    # Hessian应该是一个二阶对称矩阵，维数应该是（Natom\*3,Natom\*3）

    # In[21]:


    hessian.shape


    # 可以看到Hessian是一个维数是（Natom*3,Natom*3）的二阶对称矩阵

    # In[22]:


    hessian = hessian.reshape(3 * len(u.atoms), 3 * len(u.atoms))


    # In[23]:


    hessian = ((hessian + hessian.T) / 2).reshape((len(u.atoms), 3, len(u.atoms), 3))


    # In[24]:


    hessian = hessian/((18.897161646320724)**2)#*0.0003808798033989866 #Hartree/(Bohr2 amu)

    # In[25]:


    mass_weighted_hessian = jax.numpy.einsum("AtBs, A, B -> AtBs", hessian, 1.0/jax.numpy.sqrt(u.atoms.masses), 1.0/jax.numpy.sqrt(u.atoms.masses)).reshape(3 * len(u.atoms), 3 * len(u.atoms))

    # In[26]:


    mass_weighted_hessian


    # In[27]:


    lambda_,q=jax.numpy.linalg.eigh(mass_weighted_hessian)

    # In[28]:


    # https://docs.scipy.org/doc/scipy/reference/constants.html
    from scipy.constants import physical_constants

    E_h = physical_constants["Hartree energy"][0]
    a_0 = physical_constants["Bohr radius"][0]
    N_A = physical_constants["Avogadro constant"][0]
    c_0 = physical_constants["speed of light in vacuum"][0]
    e_c = physical_constants["elementary charge"][0]
    e_0 = physical_constants["electric constant"][0]
    mu_0 = physical_constants["mag. constant"][0]


    # In[29]:

    # 1 Hartrr = E_h J
    freq_cm_1 = jax.numpy.sqrt(jax.numpy.abs(lambda_ * E_h * 0.0003808798033989866 * 1000 * N_A / a_0**2)) / (2 * jax.numpy.pi * c_0 * 100) * ((lambda_ > 0) * 2 - 1)


    # In[30]:


    # 以上振动包含了平动和转动，应该把它们消除掉。
    # 去除平动、转动对频率的贡献，其过程大致是预先将平动、转动的模式求取，随后将力常数张量投影到平动、转动模式的补空间 ($3 n_\mathrm{Atom} - 6$ 维度空间)，得到新的力常数张量。
    # 
    # 其中的大部分内容应当在 Wilson et al.(Wilson, E. B.; Decius, J. C.; Cross, P. C. *Molecular Vibrations*; Dover Pub. Inc., 1980) 的 Chapter 2 可以找到。 

    # In[38]:


    mol_coord = u.coord.positions * 1.8897161646320724 # Bohr
    mol_weight = u.atoms.masses

    # In[39]:


    center_coord = (mol_coord * mol_weight[:, None]).sum(axis=0) / mol_weight.sum()


    # In[40]:


    center_coord


    # `centered_coord` $A^\mathrm{C}_t$ 是将质心平移至原点后的原子坐标，维度 $(n_\mathrm{Atom}, 3)$，单位 Bohr。
    #  
    # $$
    # A^\mathrm{C}_t = A_t - C_t
    # $$

    # In[41]:


    centered_coord = mol_coord - center_coord


    # ### 转动惯量本征向量
    # 
    # `rot_tmp` $I_{ts}$ 是转动惯量相关的矩阵，在初始化时维度为 $(n_\mathrm{Atom}, 3, 3)$，最终结果通过求和得到 $(3, 3)$ 的矩阵，单位 Bohr<sup>2</sup> amu。
    #  
    # $$
    # \begin{split}
    # I_{ts} =
    # \begin{cases}
    #      \sum_{A} w_A \left( - (A_t^\mathrm{C})^2 + \sum_r (A_r^\mathrm{C})^2 \right) \,, & t = s \\
    #      \sum_{A} w_A \left( - A_t^\mathrm{C} A_s^\mathrm{C} \right) \,, & t \neq s
    #  \end{cases}
    #  \end{split}
    # $$

    # In[53]:

    natm = len(u.atoms)
    rot_tmp = np.zeros((natm, 3, 3))
    rot_tmp[:, 0, 0] = centered_coord[:, 1]**2 + centered_coord[:, 2]**2
    rot_tmp[:, 1, 1] = centered_coord[:, 2]**2 + centered_coord[:, 0]**2
    rot_tmp[:, 2, 2] = centered_coord[:, 0]**2 + centered_coord[:, 1]**2
    rot_tmp[:, 0, 1] = rot_tmp[:, 1, 0] = - centered_coord[:, 0] * centered_coord[:, 1]
    rot_tmp[:, 1, 2] = rot_tmp[:, 2, 1] = - centered_coord[:, 1] * centered_coord[:, 2]
    rot_tmp[:, 2, 0] = rot_tmp[:, 0, 2] = - centered_coord[:, 2] * centered_coord[:, 0]
    rot_tmp = (rot_tmp * mol_weight[:, None, None]).sum(axis=0)


    # `rot_eig` $R_{ts}$ 是转动惯量相关的对称矩阵 $I_{ts}$ 所求得的本征向量，维度 $(3, 3)$，无量纲

    # In[54]:


    _, rot_eig = np.linalg.eigh(rot_tmp)


    # In[55]:


    rot_eig


    # ### 平动、转动投影矩阵
    # 
    # `proj_scr` $P_{A_t q}$ 是平动、转动的 $(3 n_\mathrm{Atom}, 6)$ 维度投影矩阵，其目的是将 $\Theta^{A_t B_s}$ 中不应对分子振动产生贡献的部分投影消去，剩余的 $3 n_\mathrm{Atom} - 6$ 子空间用于求取实际的分子振动频率。但在初始化 `proj_scr` $P_{A_t q}$ 时，先使用 $(n_\mathrm{Atom}, 3, 6)$ 维度的张量。
    #  
    # 在计算投影矩阵前，我们先生成 `rot_coord` $\mathscr{R}_{Asrw}$ 转动投影相关量，维度 $(n_\mathrm{Atom}, 3, 3, 3)$：
    #  
    # $$
    # \mathscr{R}_{Asrw} = \sum_{t} A^\mathrm{C}_t R_{ts} R_{rw}
    #  $$

    # In[56]:


    #rot_coord = jax.numpy.einsum("At, ts, rw -> Asrw", centered_coord, rot_eig, rot_eig)
    rot_coord = np.einsum("At, ts, rw -> Asrw", centered_coord, rot_eig, rot_eig)
    print("max_diff:",(rot_coord-rot_coord1).max())

    # 随后我们给出 `proj_scr` 的计算表达式。`proj_scr` 的前三列表示平动投影，当 $q \in (x, y, z) = (0, 1, 2)$ 时，
    # 
    # $$
    # P_{A_t q} = \sqrt{w_A} \delta_{tq}
    # $$
    # 
    # 而当 $q \in (x, y, z) = (3, 4, 5)$ 时，
    # 
    # $$
    # \begin{split}
    # P_{A_t q} = \sqrt{w_A} \times
    # \begin{cases}
    #     \mathscr{R}_{Aytz} - \mathscr{R}_{Azty} \,, & q = x \\
    #     \mathscr{R}_{Aztx} - \mathscr{R}_{Axtz} \,, & q = y \\
    #     \mathscr{R}_{Axty} - \mathscr{R}_{Aytx} \,, & q = z
    # \end{cases}
    # \end{split}
    # $$
    # 
    # 最终，我们会将 $P_{A_t q}$ 中关于 $A_t$ 的维度进行归一化，因此最终获得的 $P_{A_t q}$ 是无量纲的。

    # In[59]:


    proj_scr = np.zeros((len(u.atoms), 3, 6))
    proj_scr[:, (0, 1, 2), (0, 1, 2)] = 1
    proj_scr[:, :, 3] = (rot_coord[:, 1, :, 2] - rot_coord[:, 2, :, 1])
    proj_scr[:, :, 4] = (rot_coord[:, 2, :, 0] - rot_coord[:, 0, :, 2])
    proj_scr[:, :, 5] = (rot_coord[:, 0, :, 1] - rot_coord[:, 1, :, 0])
    proj_scr *= np.sqrt(u.atoms.masses)[:, None, None]
    proj_scr.shape = (-1, 6)
    proj_scr /= np.linalg.norm(proj_scr, axis=0)


    # In[61]:


    e_tr, _ = jax.numpy.linalg.eigh(proj_scr.T @ mass_weighted_hessian @ proj_scr)


    # ### 平动、转动投影矩阵的补空间
    # 
    # 既然我们已经得到了平动、转动的投影，那么根据矩阵的原理，相应地我们也能获得其补空间的投影。我们令 `proj_inv` $Q_{A_t q}$ 为 $P_{A_t q}$ 的补空间投影。获得补空间的大致方式是预先定义一个仅有一个分量为 $1$ 的 $(3 n_\mathrm{Atom}, )$ 维度向量，随后通过 Schmit 正交的方式给出已有投影空间的补空间向量。组合这些 Schmit 正交的向量便获得了 $Q_{A_t q}$。
    #  
    # $Q_{A_t q}$ 的维度本应当是 $(3 n_\mathrm{Atom}, 3 n_\mathrm{Atom} - 6)$ 维。但为了程序编写方便，我们先规定 `proj_inv` 是 $(3 n_\mathrm{Atom}, 3 n_\mathrm{Atom})$ 维度，并且其中的前 6 列填入 $P_{A_t q}$；在进行 Schmit 正交化后，再将前 6 列剔除。
    # 

    # In[67]:


    proj_inv = np.zeros((len(u.atoms) * 3, len(u.atoms) * 3))
    proj_inv[:, :6] = proj_scr
    cur = 6
    for i in range(0, len(u.atoms) * 3):
        vec_i = np.einsum("Ai, i -> A", proj_inv[:, :cur], proj_inv[i, :cur])
        vec_i[i] -= 1
        if np.linalg.norm(vec_i) > 1e-8:
            proj_inv[:, cur] = vec_i / np.linalg.norm(vec_i)
            cur += 1
        if cur >= len(u.atoms) * 3:
            break
    proj_inv = proj_inv[:, 6:]


    # 我们最后获得的 $Q_{A_t q}$ 是列正交切归一的矩阵，且形式大致是下三角矩阵。但需要留意，对于当前的分子，最后一列只有 6 个非零值，与倒数第二列非零值的数量相差 2 个。

    # ### 去除平动、转动部分的频率
    # 
    # 我们将对矩阵 $\mathbf{Q}^\dagger \mathbf{\Theta} \mathbf{Q}$ 进行对角化；且获得的第 $q$ 个简正坐标的频率相关量 `e` $K_q = k_q / m_q$ 与原始简正坐标 `q` $\mathbf{q}^\mathrm{orig}$ 表示如下：
    # 
    # $$
    # \mathbf{Q}^\dagger \mathbf{\Theta} \mathbf{Q} \mathbf{q}^\mathrm{orig} = \mathbf{q}^\mathrm{orig} \mathrm{diag} (\boldsymbol{K})
    # $$

    # In[96]:
    print(mass_weighted_hessian,proj_inv)
    e, q = jax.numpy.linalg.eigh(proj_inv.T @ mass_weighted_hessian @ proj_inv)
    freq_cm_1 = jax.numpy.sqrt(jax.numpy.abs(e * E_h*0.0003808798033989866 * 1000 * N_A / a_0**2)) / (2 * jax.numpy.pi * c_0 * 100) * ((e > 0) * 2 - 1)   
    print("______________________________________")
    print("JAX")
    print(e)
    print("Calculated frequency here:",freq_cm_1)    
    print("______________________________________")
    theta = np.load("theta.npy")
    proj_inv1 = np.load("proj_inv.npy")
    e, q = np.linalg.eigh(proj_inv1.T @ theta @ proj_inv1)
    print("NUMPY")
    print(e) 
    freq_cm_1 = np.sqrt(np.abs(e * E_h*0.0003808798033989866 * 1000 * N_A / a_0**2)) / (2 * np.pi * c_0 * 100) * ((e > 0) * 2 - 1)    
    print("Calculated frequency here:",freq_cm_1)    
    print("______________________________________")       
    print((proj_inv-proj_inv1).max())
    print((mass_weighted_hessian-theta).max())

    print((proj_inv.T @ mass_weighted_hessian @ proj_inv-proj_inv1.T @ theta @ proj_inv1).max())
    print(proj_inv.T @ mass_weighted_hessian @ proj_inv-proj_inv1.T @ theta @ proj_inv1)
    # 由此，我们就可以立即获得去除平动、转动部分的，以 cm<sup>-1</sup> 为单位的，总数为 $3 n_\mathrm{Atom} - 6$ 的分子频率 `freq_cm_1`：

    # In[98]:


    return freq_cm_1[0]


calculate_cm(positions, box, pairs, params)

