"""
电解质NRTL模型 (Electrolyte NRTL Model)
用于含电解质液体系统的活度系数计算

基于Chen和Evans的电解质NRTL理论
适用于水-盐-有机物三元及多元体系

作者: OpenAspen项目组
版本: 1.0.0
"""

import numpy as np
from scipy.optimize import fsolve
from typing import List, Dict, Optional, Tuple
from ..base_property_package import PropertyPackage


class ElectrolyteNRTL(PropertyPackage):
    """
    电解质NRTL模型实现
    
    特点:
    - 处理电解质溶液
    - PDH长程静电作用
    - 局部组成短程作用
    - 适用于高离子强度
    """
    
    def __init__(self, compounds: List[str], electrolytes: List[str] = None):
        """
        初始化电解质NRTL模型
        
        Parameters:
        -----------
        compounds : List[str]
            分子组分列表
        electrolytes : List[str], optional
            电解质组分列表
        """
        super().__init__(compounds, "Electrolyte NRTL")
        self.electrolytes = electrolytes or []
        self.all_species = compounds + self.electrolytes
        
        # 模型参数
        self._initialize_electrolyte_parameters()
        self._initialize_ion_properties()
    
    def _initialize_electrolyte_parameters(self):
        """初始化电解质模型参数"""
        # NRTL binary parameters (分子-分子)
        self.tau_mm = {}
        self.alpha_mm = {}
        
        # Born parameters (分子-离子)
        self.tau_mc = {}
        self.tau_ca = {}
        self.alpha_mc = {}
        self.alpha_ca = {}
        
        # 离子-离子参数
        self.tau_cc = {}
        self.tau_aa = {}
        self.alpha_cc = {}
        self.alpha_aa = {}
        
        # 默认参数设置
        self._set_default_parameters()
    
    def _initialize_ion_properties(self):
        """初始化离子性质"""
        # 离子半径 (Angstrom)
        self.ion_radii = {
            'Na+': 1.02, 'K+': 1.33, 'Ca2+': 1.00, 'Mg2+': 0.72,
            'Cl-': 1.81, 'SO42-': 2.32, 'OH-': 1.40, 'NO3-': 1.79,
            'NH4+': 1.43, 'H+': 0.0, 'HSO4-': 2.06
        }
        
        # 离子电荷数
        self.ion_charges = {
            'Na+': 1, 'K+': 1, 'Ca2+': 2, 'Mg2+': 2,
            'Cl-': -1, 'SO42-': -2, 'OH-': -1, 'NO3-': -1,
            'NH4+': 1, 'H+': 1, 'HSO4-': -1
        }
        
        # 水化数
        self.hydration_numbers = {
            'Na+': 4.0, 'K+': 6.0, 'Ca2+': 6.0, 'Mg2+': 6.0,
            'Cl-': 6.0, 'SO42-': 8.0, 'OH-': 4.0, 'NO3-': 4.0
        }
    
    def _set_default_parameters(self):
        """设置默认参数"""
        # 水-NaCl系统参数 (Chen et al., 1982)
        self.tau_mc[('H2O', 'Na+')] = 8.885
        self.tau_mc[('H2O', 'Cl-')] = -4.549
        self.alpha_mc[('H2O', 'Na+')] = 2.0
        self.alpha_mc[('H2O', 'Cl-')] = 2.0
        
        # 离子-离子参数
        self.tau_cc[('Na+', 'Ca2+')] = 0.0
        self.tau_aa[('Cl-', 'SO42-')] = 0.0
        self.alpha_cc[('Na+', 'Ca2+')] = 0.2
        self.alpha_aa[('Cl-', 'SO42-')] = 0.2
    
    def calculate_ionic_strength(self, x_ions: Dict[str, float]) -> float:
        """
        计算离子强度
        
        Parameters:
        -----------
        x_ions : Dict[str, float]
            离子摩尔分数字典
            
        Returns:
        --------
        float
            离子强度 [mol/kg]
        """
        I = 0.0
        for ion, x_ion in x_ions.items():
            if ion in self.ion_charges:
                z_ion = self.ion_charges[ion]
                I += 0.5 * x_ion * z_ion**2
        
        return I
    
    def calculate_debye_huckel_parameter(self, T: float) -> float:
        """
        计算Debye-Hückel参数
        
        Parameters:
        -----------
        T : float
            温度 [K]
            
        Returns:
        --------
        float
            Debye-Hückel参数 A_phi
        """
        # 水的介电常数 (Bradley & Pitzer, 1979)
        epsilon_r = 78.54 * (1 - 4.579e-3 * (T - 298.15) + 
                             1.19e-5 * (T - 298.15)**2)
        
        # Debye-Hückel参数
        A_phi = 1.4006e6 * (1000)**0.5 / (epsilon_r * T)**1.5
        
        return A_phi
    
    def calculate_pdh_activity_coefficient(self, ion: str, I: float, 
                                         T: float) -> float:
        """
        计算Pitzer-Debye-Hückel长程贡献
        
        Parameters:
        -----------
        ion : str
            离子名称
        I : float
            离子强度 [mol/kg]
        T : float
            温度 [K]
            
        Returns:
        --------
        float
            PDH活度系数
        """
        if ion not in self.ion_charges:
            return 1.0
        
        z_ion = abs(self.ion_charges[ion])
        A_phi = self.calculate_debye_huckel_parameter(T)
        
        # Pitzer方程
        sqrt_I = np.sqrt(I)
        
        ln_gamma_pdh = -A_phi * z_ion**2 * sqrt_I / (1 + 1.2 * sqrt_I)
        
        return np.exp(ln_gamma_pdh)
    
    def calculate_local_composition_contribution(self, species: str, 
                                               x: Dict[str, float], 
                                               T: float) -> float:
        """
        计算局部组成贡献
        
        Parameters:
        -----------
        species : str
            物种名称
        x : Dict[str, float]
            摩尔分数字典
        T : float
            温度 [K]
            
        Returns:
        --------
        float
            局部组成活度系数
        """
        ln_gamma_lc = 0.0
        
        # 分子的局部组成贡献
        if species in self.compounds:
            ln_gamma_lc = self._calculate_molecular_lc_contribution(species, x, T)
        
        # 离子的局部组成贡献
        elif species in self.electrolytes:
            ln_gamma_lc = self._calculate_ionic_lc_contribution(species, x, T)
        
        return np.exp(ln_gamma_lc)
    
    def _calculate_molecular_lc_contribution(self, molecule: str, 
                                           x: Dict[str, float], 
                                           T: float) -> float:
        """
        计算分子的局部组成贡献
        """
        sum1 = 0.0
        sum2 = 0.0
        
        # 第一项：分子-分子相互作用
        for j in self.compounds:
            if j in x:
                tau_ij = self.tau_mm.get((molecule, j), 0.0)
                alpha_ij = self.alpha_mm.get((molecule, j), 0.2)
                G_ij = np.exp(-alpha_ij * tau_ij)
                
                sum1 += x[j] * tau_ij * G_ij
                sum2 += x[j] * G_ij
        
        # 第二项：分子-离子相互作用
        for ion in self.electrolytes:
            if ion in x:
                if ion in self.ion_charges and self.ion_charges[ion] > 0:  # 阳离子
                    tau_ic = self.tau_mc.get((molecule, ion), 0.0)
                    alpha_ic = self.alpha_mc.get((molecule, ion), 2.0)
                else:  # 阴离子
                    tau_ic = self.tau_ca.get((molecule, ion), 0.0)
                    alpha_ic = self.alpha_ca.get((molecule, ion), 2.0)
                
                G_ic = np.exp(-alpha_ic * tau_ic)
                sum1 += x[ion] * tau_ic * G_ic
                sum2 += x[ion] * G_ic
        
        if sum2 > 0:
            ln_gamma_lc = sum1 / sum2
        else:
            ln_gamma_lc = 0.0
        
        return ln_gamma_lc
    
    def _calculate_ionic_lc_contribution(self, ion: str, 
                                       x: Dict[str, float], 
                                       T: float) -> float:
        """
        计算离子的局部组成贡献
        """
        ln_gamma_lc = 0.0
        
        # 离子-分子相互作用
        for molecule in self.compounds:
            if molecule in x:
                if self.ion_charges.get(ion, 0) > 0:  # 阳离子
                    tau_cm = self.tau_mc.get((molecule, ion), 0.0)
                    alpha_cm = self.alpha_mc.get((molecule, ion), 2.0)
                else:  # 阴离子
                    tau_am = self.tau_ca.get((molecule, ion), 0.0)
                    alpha_am = self.alpha_ca.get((molecule, ion), 2.0)
                
                G_cm = np.exp(-alpha_cm * tau_cm)
                ln_gamma_lc += x[molecule] * tau_cm * G_cm
        
        return ln_gamma_lc
    
    def calculate_activity_coefficient(self, species: str, x: Dict[str, float], 
                                     T: float) -> float:
        """
        计算总活度系数
        
        Parameters:
        -----------
        species : str
            物种名称
        x : Dict[str, float]
            摩尔分数字典
        T : float
            温度 [K]
            
        Returns:
        --------
        float
            活度系数
        """
        # 提取离子摩尔分数
        x_ions = {ion: x.get(ion, 0.0) for ion in self.electrolytes}
        
        # 计算离子强度
        I = self.calculate_ionic_strength(x_ions)
        
        # PDH长程贡献
        gamma_pdh = self.calculate_pdh_activity_coefficient(species, I, T)
        
        # 局部组成短程贡献
        gamma_lc = self.calculate_local_composition_contribution(species, x, T)
        
        # 总活度系数
        gamma_total = gamma_pdh * gamma_lc
        
        return gamma_total
    
    def calculate_activity_coefficients(self, x: np.ndarray, T: float) -> np.ndarray:
        """
        计算所有组分的活度系数
        
        Parameters:
        -----------
        x : np.ndarray
            摩尔分数数组
        T : float
            温度 [K]
            
        Returns:
        --------
        np.ndarray
            活度系数数组
        """
        # 构建摩尔分数字典
        x_dict = {}
        for i, species in enumerate(self.all_species):
            if i < len(x):
                x_dict[species] = x[i]
        
        # 计算各组分活度系数
        gamma = np.zeros(len(self.all_species))
        for i, species in enumerate(self.all_species):
            gamma[i] = self.calculate_activity_coefficient(species, x_dict, T)
        
        return gamma[:len(x)]
    
    def set_binary_parameters(self, comp1: str, comp2: str, 
                             tau12: float, tau21: float, 
                             alpha12: float = 0.2):
        """
        设置二元交互参数
        
        Parameters:
        -----------
        comp1, comp2 : str
            组分名称
        tau12, tau21 : float
            二元交互参数
        alpha12 : float
            非随机参数
        """
        # 分子-分子参数
        if comp1 in self.compounds and comp2 in self.compounds:
            self.tau_mm[(comp1, comp2)] = tau12
            self.tau_mm[(comp2, comp1)] = tau21
            self.alpha_mm[(comp1, comp2)] = alpha12
            self.alpha_mm[(comp2, comp1)] = alpha12
        
        # 分子-离子参数
        elif comp1 in self.compounds and comp2 in self.electrolytes:
            if self.ion_charges.get(comp2, 0) > 0:  # 阳离子
                self.tau_mc[(comp1, comp2)] = tau12
                self.alpha_mc[(comp1, comp2)] = alpha12
            else:  # 阴离子
                self.tau_ca[(comp1, comp2)] = tau12
                self.alpha_ca[(comp1, comp2)] = alpha12
    
    def set_ion_properties(self, ion: str, charge: int, 
                          radius: float = None, 
                          hydration_number: float = None):
        """
        设置离子性质
        
        Parameters:
        -----------
        ion : str
            离子名称
        charge : int
            电荷数
        radius : float, optional
            离子半径 [Angstrom]
        hydration_number : float, optional
            水化数
        """
        self.ion_charges[ion] = charge
        
        if radius is not None:
            self.ion_radii[ion] = radius
        
        if hydration_number is not None:
            self.hydration_numbers[ion] = hydration_number
    
    def calculate_osmotic_coefficient(self, x_solvent: float, x_ions: Dict[str, float], 
                                    T: float) -> float:
        """
        计算渗透系数
        
        Parameters:
        -----------
        x_solvent : float
            溶剂摩尔分数
        x_ions : Dict[str, float]
            离子摩尔分数字典
        T : float
            温度 [K]
            
        Returns:
        --------
        float
            渗透系数
        """
        # 构建完整摩尔分数字典
        x_dict = {'H2O': x_solvent}
        x_dict.update(x_ions)
        
        # 计算活度系数
        gamma_w = self.calculate_activity_coefficient('H2O', x_dict, T)
        
        # 计算渗透系数
        sum_x_ions = sum(x_ions.values())
        if sum_x_ions > 0:
            phi = -np.log(gamma_w * x_solvent) / sum_x_ions
        else:
            phi = 1.0
        
        return phi
    
    def get_model_info(self) -> Dict[str, any]:
        """
        获取模型信息
        
        Returns:
        --------
        Dict[str, any]
            模型信息字典
        """
        return {
            'name': self.name,
            'type': 'Electrolyte Activity Coefficient Model',
            'compounds': self.compounds,
            'electrolytes': self.electrolytes,
            'applicable_range': {
                'temperature': '273-373 K',
                'ionic_strength': '0-20 mol/kg',
                'phases': ['liquid'],
                'systems': ['aqueous electrolytes', 'mixed solvents with salts']
            },
            'theory': ['Pitzer-Debye-Hückel', 'Local Composition', 'NRTL'],
            'limitations': [
                '需要可靠的二元参数',
                '高温下精度可能下降',
                '不适用于熔盐体系'
            ]
        }


# 使用示例
if __name__ == "__main__":
    # 创建电解质NRTL模型
    compounds = ['H2O', 'methanol']
    electrolytes = ['Na+', 'Cl-']
    
    enrtl = ElectrolyteNRTL(compounds, electrolytes)
    
    # 设置离子性质
    enrtl.set_ion_properties('Na+', 1, 1.02, 4.0)
    enrtl.set_ion_properties('Cl-', -1, 1.81, 6.0)
    
    # 设置二元参数
    enrtl.set_binary_parameters('H2O', 'Na+', 8.885, -4.549, 2.0)
    enrtl.set_binary_parameters('H2O', 'Cl-', -4.549, 8.885, 2.0)
    
    # 计算活度系数
    x = np.array([0.8, 0.1, 0.08, 0.02])  # H2O, methanol, Na+, Cl-
    T = 298.15  # K
    
    gamma = enrtl.calculate_activity_coefficients(x, T)
    
    print("电解质NRTL模型计算结果:")
    for i, species in enumerate(enrtl.all_species[:len(x)]):
        print(f"{species}: γ = {gamma[i]:.6f}")
    
    # 计算离子强度
    x_ions = {'Na+': x[2], 'Cl-': x[3]}
    I = enrtl.calculate_ionic_strength(x_ions)
    print(f"\n离子强度: {I:.6f} mol/kg")
    
    # 计算渗透系数
    phi = enrtl.calculate_osmotic_coefficient(x[0], x_ions, T)
    print(f"渗透系数: {phi:.6f}")
    
    # 获取模型信息
    model_info = enrtl.get_model_info()
    print(f"\n模型信息: {model_info['name']}")
    print(f"适用范围: {model_info['applicable_range']}") 