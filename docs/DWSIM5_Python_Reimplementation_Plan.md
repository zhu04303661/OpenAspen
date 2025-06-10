# DWSIM5 Python重新实现方案

## 1. 项目概述

基于DWSIM5的.NET架构，使用Python重新实现核心业务逻辑，形成现代化的化工过程仿真API服务。

## 2. Python版本架构设计

### 2.1 整体项目结构

```
dwsim-python/
├── dwsim_core/                     # 核心架构层
│   ├── __init__.py
│   ├── interfaces/                 # 接口定义
│   │   ├── __init__.py
│   │   ├── base_interfaces.py      # 基础接口
│   │   ├── property_package.py     # 物性包接口
│   │   ├── simulation_object.py    # 仿真对象接口
│   │   └── flash_algorithm.py      # 闪蒸算法接口
│   ├── exceptions/                 # 异常处理
│   │   ├── __init__.py
│   │   ├── thermodynamic_errors.py
│   │   └── convergence_errors.py
│   └── utilities/                  # 工具类
│       ├── __init__.py
│       ├── unit_converter.py       # 单位转换
│       ├── constants.py            # 物理常数
│       └── math_utils.py           # 数学工具
├── dwsim_data/                     # 数据访问层
│   ├── __init__.py
│   ├── databases/                  # 数据库管理
│   │   ├── __init__.py
│   │   ├── component_db.py         # 组分数据库
│   │   ├── interaction_params.py   # 交互参数
│   │   └── database_manager.py     # 数据库管理器
│   ├── loaders/                    # 数据加载器
│   │   ├── __init__.py
│   │   ├── xml_loader.py           # XML数据加载
│   │   ├── json_loader.py          # JSON数据加载
│   │   └── text_loader.py          # 文本数据加载
│   └── assets/                     # 数据文件
│       ├── databases/
│       │   ├── dwsim_components.json
│       │   ├── chemsep_data.json
│       │   └── coolprop_data.json
│       └── parameters/
│           ├── unifac_groups.json
│           ├── nrtl_params.json
│           └── eos_parameters.json
├── dwsim_thermo/                   # 计算引擎层
│   ├── __init__.py
│   ├── property_packages/          # 物性包实现
│   │   ├── __init__.py
│   │   ├── base_package.py         # 基础物性包
│   │   ├── eos_packages/           # 状态方程物性包
│   │   │   ├── __init__.py
│   │   │   ├── peng_robinson.py
│   │   │   ├── srk.py
│   │   │   └── lee_kesler.py
│   │   ├── activity_packages/      # 活度系数物性包
│   │   │   ├── __init__.py
│   │   │   ├── unifac.py
│   │   │   ├── nrtl.py
│   │   │   └── uniquac.py
│   │   └── specialized/            # 特殊物性包
│   │       ├── __init__.py
│   │       ├── coolprop_wrapper.py
│   │       └── steam_tables.py
│   ├── flash_algorithms/           # 闪蒸算法
│   │   ├── __init__.py
│   │   ├── base_flash.py
│   │   ├── pt_flash.py             # PT闪蒸
│   │   ├── ph_flash.py             # PH闪蒸
│   │   └── ps_flash.py             # PS闪蒸
│   ├── equations/                  # 状态方程
│   │   ├── __init__.py
│   │   ├── cubic_eos.py            # 立方状态方程
│   │   ├── activity_models.py      # 活度系数模型
│   │   └── mixing_rules.py         # 混合规则
│   └── solvers/                    # 数值求解器
│       ├── __init__.py
│       ├── newton_raphson.py
│       ├── successive_substitution.py
│       └── optimization.py
├── dwsim_operations/               # 业务逻辑层
│   ├── __init__.py
│   ├── unit_operations/            # 单元操作
│   │   ├── __init__.py
│   │   ├── base_unit.py            # 基础单元操作
│   │   ├── separators/             # 分离设备
│   │   │   ├── __init__.py
│   │   │   ├── flash_vessel.py
│   │   │   ├── distillation.py
│   │   │   └── absorption.py
│   │   ├── heat_exchangers/        # 换热设备
│   │   │   ├── __init__.py
│   │   │   ├── heater_cooler.py
│   │   │   └── heat_exchanger.py
│   │   └── reactors/               # 反应器
│   │       ├── __init__.py
│   │       ├── cstr.py
│   │       └── pfr.py
│   ├── flowsheet/                  # 流程图
│   │   ├── __init__.py
│   │   ├── flowsheet.py            # 流程图类
│   │   ├── stream.py               # 物流类
│   │   └── solver.py               # 流程图求解器
│   └── optimization/               # 优化
│       ├── __init__.py
│       ├── sensitivity.py          # 灵敏度分析
│       └── optimizer.py            # 优化器
├── dwsim_api/                      # API服务层
│   ├── __init__.py
│   ├── main.py                     # FastAPI主程序
│   ├── routers/                    # API路由
│   │   ├── __init__.py
│   │   ├── properties.py           # 物性计算API
│   │   ├── flash.py                # 闪蒸计算API
│   │   ├── simulation.py           # 仿真计算API
│   │   └── components.py           # 组分管理API
│   ├── models/                     # Pydantic模型
│   │   ├── __init__.py
│   │   ├── component_models.py
│   │   ├── calculation_models.py
│   │   └── response_models.py
│   ├── services/                   # 服务层
│   │   ├── __init__.py
│   │   ├── property_service.py
│   │   ├── flash_service.py
│   │   └── simulation_service.py
│   └── middleware/                 # 中间件
│       ├── __init__.py
│       ├── error_handler.py
│       └── validation.py
├── tests/                          # 测试
│   ├── __init__.py
│   ├── test_thermo/
│   ├── test_operations/
│   ├── test_api/
│   └── benchmarks/
├── docs/                           # 文档
│   ├── api_docs/
│   ├── user_guide/
│   └── developer_guide/
├── requirements.txt                # 依赖管理
├── pyproject.toml                 # 项目配置
├── docker-compose.yml             # Docker配置
└── README.md                      # 项目说明
```

## 3. 核心组件Python实现

### 3.1 核心架构层实现

#### 基础接口定义 (dwsim_core/interfaces/base_interfaces.py)

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from enum import Enum
import numpy as np

class PhaseType(Enum):
    VAPOR = "vapor"
    LIQUID = "liquid"
    SOLID = "solid"
    AQUEOUS = "aqueous"

class PropertyType(Enum):
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    ENTHALPY = "enthalpy"
    ENTROPY = "entropy"
    DENSITY = "density"
    VISCOSITY = "viscosity"
    THERMAL_CONDUCTIVITY = "thermal_conductivity"

class IComponent(ABC):
    """组分接口"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def cas_number(self) -> str:
        pass
    
    @property
    @abstractmethod
    def molecular_weight(self) -> float:
        pass
    
    @property
    @abstractmethod
    def critical_temperature(self) -> float:
        pass
    
    @property
    @abstractmethod
    def critical_pressure(self) -> float:
        pass

class IStream(ABC):
    """物流接口"""
    
    @abstractmethod
    def get_property(self, property_type: PropertyType, phase: PhaseType = None) -> float:
        pass
    
    @abstractmethod
    def set_property(self, property_type: PropertyType, value: float, phase: PhaseType = None):
        pass
    
    @abstractmethod
    def get_composition(self, phase: PhaseType = None) -> Dict[str, float]:
        pass

class IPropertyPackage(ABC):
    """物性包接口"""
    
    @abstractmethod
    def calculate_properties(self, stream: IStream, properties: List[PropertyType]) -> Dict[PropertyType, float]:
        pass
    
    @abstractmethod
    def flash_calculation(self, stream: IStream, spec1: PropertyType, spec2: PropertyType) -> IStream:
        pass

class IUnitOperation(ABC):
    """单元操作接口"""
    
    @abstractmethod
    def calculate(self) -> bool:
        pass
    
    @abstractmethod
    def get_inlet_streams(self) -> List[IStream]:
        pass
    
    @abstractmethod
    def get_outlet_streams(self) -> List[IStream]:
        pass
```

#### 物性包基类 (dwsim_thermo/property_packages/base_package.py)

```python
from dwsim_core.interfaces.base_interfaces import IPropertyPackage, IStream, PropertyType, PhaseType
from dwsim_data.databases.component_db import ComponentDatabase
from typing import Dict, List
import numpy as np

class BasePropertyPackage(IPropertyPackage):
    """物性包基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.component_db = ComponentDatabase()
        self._components = {}
        
    def add_component(self, component_name: str):
        """添加组分"""
        component_data = self.component_db.get_component(component_name)
        if component_data:
            self._components[component_name] = component_data
        else:
            raise ValueError(f"Component {component_name} not found in database")
    
    def calculate_vapor_pressure(self, component: str, temperature: float) -> float:
        """计算蒸汽压 - Antoine方程"""
        comp_data = self._components[component]
        A, B, C = comp_data['antoine_coefficients']
        return 10**(A - B/(temperature + C))
    
    def calculate_ideal_gas_cp(self, component: str, temperature: float) -> float:
        """计算理想气体热容"""
        comp_data = self._components[component]
        coeffs = comp_data['cp_coefficients']
        T = temperature
        return coeffs[0] + coeffs[1]*T + coeffs[2]*T**2 + coeffs[3]*T**3
    
    def calculate_mixture_property(self, property_values: Dict[str, float], 
                                 composition: Dict[str, float], 
                                 mixing_rule: str = "linear") -> float:
        """混合物性质计算"""
        if mixing_rule == "linear":
            return sum(composition[comp] * property_values[comp] 
                      for comp in composition.keys())
        else:
            raise NotImplementedError(f"Mixing rule {mixing_rule} not implemented")
```

### 3.2 数据访问层实现

#### 组分数据库 (dwsim_data/databases/component_db.py)

```python
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from dwsim_data.loaders.json_loader import JsonLoader

class ComponentDatabase:
    """组分数据库管理器"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self._get_default_db_path()
        self.json_loader = JsonLoader()
        self._components_cache = {}
        self._load_components()
    
    def _get_default_db_path(self) -> str:
        """获取默认数据库路径"""
        current_dir = Path(__file__).parent
        return str(current_dir / "../assets/databases/dwsim_components.json")
    
    def _load_components(self):
        """加载组分数据"""
        try:
            components_data = self.json_loader.load_json(self.db_path)
            for comp in components_data['components']:
                self._components_cache[comp['name']] = comp
                # 同时以CAS号为键存储
                if 'cas_number' in comp:
                    self._components_cache[comp['cas_number']] = comp
        except Exception as e:
            print(f"Error loading components database: {e}")
    
    def get_component(self, identifier: str) -> Optional[Dict]:
        """根据名称或CAS号获取组分数据"""
        return self._components_cache.get(identifier)
    
    def search_components(self, keyword: str) -> List[Dict]:
        """搜索组分"""
        results = []
        keyword_lower = keyword.lower()
        for comp in self._components_cache.values():
            if (keyword_lower in comp.get('name', '').lower() or 
                keyword_lower in comp.get('formula', '').lower()):
                results.append(comp)
        return results
    
    def get_all_components(self) -> List[Dict]:
        """获取所有组分"""
        # 去重，因为可能同时以name和cas_number为键存储
        unique_components = {}
        for comp in self._components_cache.values():
            unique_components[comp['name']] = comp
        return list(unique_components.values())
```

#### 交互参数管理 (dwsim_data/databases/interaction_params.py)

```python
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

class InteractionParameterDB:
    """交互参数数据库"""
    
    def __init__(self):
        self.unifac_params = self._load_unifac_params()
        self.nrtl_params = self._load_nrtl_params()
        self.pr_params = self._load_pr_params()
    
    def _load_unifac_params(self) -> Dict:
        """加载UNIFAC参数"""
        params_path = Path(__file__).parent / "../assets/parameters/unifac_groups.json"
        with open(params_path, 'r') as f:
            return json.load(f)
    
    def _load_nrtl_params(self) -> Dict:
        """加载NRTL参数"""
        params_path = Path(__file__).parent / "../assets/parameters/nrtl_params.json"
        with open(params_path, 'r') as f:
            return json.load(f)
    
    def _load_pr_params(self) -> Dict:
        """加载Peng-Robinson交互参数"""
        params_path = Path(__file__).parent / "../assets/parameters/eos_parameters.json"
        with open(params_path, 'r') as f:
            return json.load(f)
    
    def get_unifac_groups(self, component: str) -> Dict[int, int]:
        """获取组分的UNIFAC基团"""
        return self.unifac_params.get('components', {}).get(component, {})
    
    def get_unifac_interaction(self, group1: int, group2: int) -> Tuple[float, float]:
        """获取UNIFAC基团交互参数"""
        key1 = f"{group1}-{group2}"
        key2 = f"{group2}-{group1}"
        
        params = self.unifac_params.get('interactions', {})
        if key1 in params:
            return params[key1]['a12'], params[key1]['a21']
        elif key2 in params:
            return params[key2]['a21'], params[key2]['a12']
        else:
            return 0.0, 0.0  # 默认值
    
    def get_nrtl_params(self, comp1: str, comp2: str) -> Tuple[float, float, float]:
        """获取NRTL参数 (A12, A21, alpha)"""
        pair_key = f"{comp1}-{comp2}"
        reverse_key = f"{comp2}-{comp1}"
        
        params = self.nrtl_params.get('binary_params', {})
        if pair_key in params:
            p = params[pair_key]
            return p['A12'], p['A21'], p['alpha']
        elif reverse_key in params:
            p = params[reverse_key]
            return p['A21'], p['A12'], p['alpha']
        else:
            return 0.0, 0.0, 0.3  # 默认值
    
    def get_pr_kij(self, comp1: str, comp2: str) -> float:
        """获取PR状态方程交互参数"""
        pair_key = f"{comp1}-{comp2}"
        reverse_key = f"{comp2}-{comp1}"
        
        params = self.pr_params.get('binary_interaction_parameters', {})
        if pair_key in params:
            return params[pair_key]
        elif reverse_key in params:
            return params[reverse_key]
        else:
            return 0.0  # 默认值
```

### 3.3 计算引擎层实现

#### Peng-Robinson状态方程 (dwsim_thermo/property_packages/eos_packages/peng_robinson.py)

```python
import numpy as np
from scipy.optimize import fsolve
from dwsim_thermo.property_packages.base_package import BasePropertyPackage
from dwsim_core.interfaces.base_interfaces import IStream, PropertyType, PhaseType
from typing import Dict, List, Tuple

class PengRobinsonPackage(BasePropertyPackage):
    """Peng-Robinson状态方程物性包"""
    
    def __init__(self):
        super().__init__("Peng-Robinson")
        self.R = 8.314  # 气体常数
    
    def calculate_pr_parameters(self, component: str, temperature: float) -> Tuple[float, float]:
        """计算PR方程参数a和b"""
        comp_data = self._components[component]
        Tc = comp_data['critical_temperature']
        Pc = comp_data['critical_pressure']
        omega = comp_data['acentric_factor']
        
        # 计算a和b参数
        a0 = 0.45724 * (self.R * Tc)**2 / Pc
        b = 0.07780 * self.R * Tc / Pc
        
        # 温度修正函数
        Tr = temperature / Tc
        kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
        if omega > 0.491:
            kappa = 0.379642 + 1.48503*omega - 0.164423*omega**2 + 0.016666*omega**3
        
        alpha = (1 + kappa*(1 - np.sqrt(Tr)))**2
        a = a0 * alpha
        
        return a, b
    
    def calculate_fugacity_coefficient(self, composition: Dict[str, float], 
                                    temperature: float, pressure: float,
                                    phase: PhaseType) -> Dict[str, float]:
        """计算逸度系数"""
        components = list(composition.keys())
        n_comp = len(components)
        
        # 计算纯组分参数
        a_pure = {}
        b_pure = {}
        for comp in components:
            a_pure[comp], b_pure[comp] = self.calculate_pr_parameters(comp, temperature)
        
        # 计算混合物参数
        b_mix = sum(composition[comp] * b_pure[comp] for comp in components)
        
        a_mix = 0.0
        for i, comp_i in enumerate(components):
            for j, comp_j in enumerate(components):
                kij = self._get_interaction_parameter(comp_i, comp_j)
                aij = np.sqrt(a_pure[comp_i] * a_pure[comp_j]) * (1 - kij)
                a_mix += composition[comp_i] * composition[comp_j] * aij
        
        # 计算压缩因子
        A = a_mix * pressure / (self.R * temperature)**2
        B = b_mix * pressure / (self.R * temperature)
        
        # 求解立方方程
        coeffs = [1, -(1-B), A-3*B**2-2*B, -(A*B-B**2-B**3)]
        z_roots = np.roots(coeffs)
        z_roots = np.real(z_roots[np.isreal(z_roots)])
        
        if phase == PhaseType.VAPOR:
            Z = np.max(z_roots)
        else:
            Z = np.min(z_roots)
        
        # 计算逸度系数
        phi = {}
        for comp in components:
            # 计算偏摩尔体积相关项
            dAdni = 2 * sum(composition[comp_j] * np.sqrt(a_pure[comp] * a_pure[comp_j]) * 
                           (1 - self._get_interaction_parameter(comp, comp_j)) 
                           for comp_j in components)
            
            ln_phi = (b_pure[comp]/b_mix * (Z-1) - np.log(Z-B) - 
                     A/(2*np.sqrt(2)*B) * (dAdni/a_mix - b_pure[comp]/b_mix) * 
                     np.log((Z + (1+np.sqrt(2))*B)/(Z + (1-np.sqrt(2))*B)))
            
            phi[comp] = np.exp(ln_phi)
        
        return phi
    
    def _get_interaction_parameter(self, comp1: str, comp2: str) -> float:
        """获取二元交互参数"""
        # 从数据库获取kij参数
        from dwsim_data.databases.interaction_params import InteractionParameterDB
        param_db = InteractionParameterDB()
        return param_db.get_pr_kij(comp1, comp2)
    
    def flash_calculation(self, stream: IStream, spec1: PropertyType, spec2: PropertyType) -> IStream:
        """PT闪蒸计算"""
        if spec1 == PropertyType.PRESSURE and spec2 == PropertyType.TEMPERATURE:
            return self._pt_flash(stream)
        else:
            raise NotImplementedError(f"Flash calculation for {spec1}-{spec2} not implemented")
    
    def _pt_flash(self, stream: IStream) -> IStream:
        """PT闪蒸"""
        temperature = stream.get_property(PropertyType.TEMPERATURE)
        pressure = stream.get_property(PropertyType.PRESSURE)
        composition = stream.get_composition()
        
        # 初始化K值
        K_values = {}
        for comp in composition.keys():
            vapor_pressure = self.calculate_vapor_pressure(comp, temperature)
            K_values[comp] = vapor_pressure / pressure
        
        # Rachford-Rice方程求解
        def rachford_rice(V):
            return sum((K_values[comp] - 1) * composition[comp] / 
                      (1 + V * (K_values[comp] - 1)) 
                      for comp in composition.keys())
        
        # 求解汽相摩尔分数
        V_frac = fsolve(rachford_rice, 0.5)[0]
        V_frac = max(0, min(1, V_frac))  # 限制在[0,1]范围内
        
        # 计算各相组成
        x_liquid = {}  # 液相组成
        y_vapor = {}   # 汽相组成
        
        for comp in composition.keys():
            x_liquid[comp] = composition[comp] / (1 + V_frac * (K_values[comp] - 1))
            y_vapor[comp] = K_values[comp] * x_liquid[comp]
        
        # 更新物流信息
        stream.set_property("vapor_fraction", V_frac)
        stream.set_composition(x_liquid, PhaseType.LIQUID)
        stream.set_composition(y_vapor, PhaseType.VAPOR)
        
        return stream
```

### 3.4 API服务层实现

#### FastAPI主程序 (dwsim_api/main.py)

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dwsim_api.routers import properties, flash, simulation, components
from dwsim_api.middleware.error_handler import setup_exception_handlers
import uvicorn

# 创建FastAPI应用
app = FastAPI(
    title="DWSIM Python API",
    description="化工过程仿真Python API服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 设置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(properties.router, prefix="/api/v1/properties", tags=["Properties"])
app.include_router(flash.router, prefix="/api/v1/flash", tags=["Flash Calculations"])
app.include_router(simulation.router, prefix="/api/v1/simulation", tags=["Simulation"])
app.include_router(components.router, prefix="/api/v1/components", tags=["Components"])

# 设置异常处理
setup_exception_handlers(app)

@app.get("/")
async def root():
    return {
        "message": "DWSIM Python API Service",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

#### 物性计算API (dwsim_api/routers/properties.py)

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from dwsim_api.services.property_service import PropertyService
from dwsim_api.models.calculation_models import PropertyRequest, PropertyResponse

router = APIRouter()
property_service = PropertyService()

class ComponentPropertyRequest(BaseModel):
    component: str
    temperature: float  # K
    pressure: float     # Pa
    property_types: List[str]
    property_package: str = "Peng-Robinson"

class MixturePropertyRequest(BaseModel):
    composition: Dict[str, float]  # 组分摩尔分数
    temperature: float  # K
    pressure: float     # Pa
    property_types: List[str]
    property_package: str = "Peng-Robinson"

@router.post("/component", response_model=PropertyResponse)
async def calculate_component_properties(request: ComponentPropertyRequest):
    """计算纯组分物性"""
    try:
        result = await property_service.calculate_component_properties(
            component=request.component,
            temperature=request.temperature,
            pressure=request.pressure,
            property_types=request.property_types,
            property_package=request.property_package
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/mixture", response_model=PropertyResponse)
async def calculate_mixture_properties(request: MixturePropertyRequest):
    """计算混合物物性"""
    try:
        # 验证组成和为1
        total_composition = sum(request.composition.values())
        if abs(total_composition - 1.0) > 1e-6:
            raise ValueError("Composition sum must equal 1.0")
        
        result = await property_service.calculate_mixture_properties(
            composition=request.composition,
            temperature=request.temperature,
            pressure=request.pressure,
            property_types=request.property_types,
            property_package=request.property_package
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/available-packages")
async def get_available_property_packages():
    """获取可用的物性包列表"""
    return {
        "property_packages": [
            "Peng-Robinson",
            "SRK",
            "UNIFAC",
            "NRTL",
            "CoolProp",
            "Ideal"
        ]
    }

@router.get("/available-properties")
async def get_available_properties():
    """获取可计算的物性列表"""
    return {
        "properties": [
            "temperature",
            "pressure", 
            "enthalpy",
            "entropy",
            "density",
            "viscosity",
            "thermal_conductivity",
            "vapor_pressure",
            "fugacity_coefficient",
            "activity_coefficient",
            "compressibility_factor"
        ]
    }
```

#### 闪蒸计算API (dwsim_api/routers/flash.py)

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from dwsim_api.services.flash_service import FlashService

router = APIRouter()
flash_service = FlashService()

class FlashRequest(BaseModel):
    composition: Dict[str, float]  # 组分摩尔分数
    specification_1: Dict[str, float]  # {"type": "temperature", "value": 298.15}
    specification_2: Dict[str, float]  # {"type": "pressure", "value": 101325}
    property_package: str = "Peng-Robinson"

class FlashResponse(BaseModel):
    vapor_fraction: float
    vapor_composition: Dict[str, float]
    liquid_composition: Dict[str, float]
    k_values: Dict[str, float]
    properties: Dict[str, Dict[str, float]]  # properties by phase

@router.post("/pt-flash", response_model=FlashResponse)
async def pt_flash_calculation(request: FlashRequest):
    """PT闪蒸计算"""
    try:
        # 验证规定
        if (request.specification_1.get("type") not in ["temperature", "pressure"] or
            request.specification_2.get("type") not in ["temperature", "pressure"]):
            raise ValueError("PT flash requires temperature and pressure specifications")
        
        result = await flash_service.pt_flash(
            composition=request.composition,
            temperature=request.specification_1.get("value") if request.specification_1.get("type") == "temperature" 
                       else request.specification_2.get("value"),
            pressure=request.specification_1.get("value") if request.specification_1.get("type") == "pressure"
                    else request.specification_2.get("value"),
            property_package=request.property_package
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/ph-flash", response_model=FlashResponse) 
async def ph_flash_calculation(request: FlashRequest):
    """PH闪蒸计算"""
    try:
        result = await flash_service.ph_flash(
            composition=request.composition,
            pressure=request.specification_1.get("value") if request.specification_1.get("type") == "pressure"
                    else request.specification_2.get("value"),
            enthalpy=request.specification_1.get("value") if request.specification_1.get("type") == "enthalpy"
                    else request.specification_2.get("value"),
            property_package=request.property_package
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/available-flash-types")
async def get_available_flash_types():
    """获取可用的闪蒸类型"""
    return {
        "flash_types": [
            "PT-Flash",  # 温度-压力闪蒸
            "PH-Flash",  # 压力-焓闪蒸
            "PS-Flash",  # 压力-熵闪蒸
            "TV-Flash",  # 温度-体积闪蒸
            "PV-Flash"   # 压力-体积闪蒸
        ]
    }
```

## 4. 部署和配置

### 4.1 Docker配置 (docker-compose.yml)

```yaml
version: '3.8'

services:
  dwsim-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - LOG_LEVEL=info
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
    
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: dwsim
      POSTGRES_USER: dwsim
      POSTGRES_PASSWORD: dwsim123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

### 4.2 依赖管理 (requirements.txt)

```
# Web框架
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0

# 科学计算
numpy==1.24.3
scipy==1.11.4
pandas==2.0.3

# 热力学库
CoolProp==6.4.1
thermo==0.2.20

# 数据库
sqlalchemy==2.0.23
alembic==1.12.1
psycopg2-binary==2.9.9

# 缓存
redis==5.0.1

# 测试
pytest==7.4.3
pytest-asyncio==0.21.1

# 工具
pyyaml==6.0.1
python-multipart==0.0.6
```

## 5. 关键技术特性

### 5.1 性能优化
- **并行计算**: 使用NumPy向量化操作
- **缓存机制**: Redis缓存频繁计算结果
- **异步处理**: FastAPI异步API处理
- **数据库优化**: SQLAlchemy ORM优化

### 5.2 扩展性设计
- **插件架构**: 支持自定义物性包
- **接口标准**: 完善的抽象接口定义
- **模块化**: 高度解耦的模块设计
- **配置化**: 灵活的配置管理

### 5.3 可靠性保证
- **错误处理**: 完善的异常处理机制
- **数据验证**: Pydantic数据验证
- **单元测试**: 全面的测试覆盖
- **日志记录**: 详细的操作日志

## 6. 迁移路径

### 6.1 第一阶段：核心功能
- 基础物性计算
- PT闪蒸算法
- 主要物性包(PR, SRK)
- 基础API接口

### 6.2 第二阶段：高级功能
- 复杂闪蒸算法
- 更多物性包
- 单元操作模块
- 流程图求解

### 6.3 第三阶段：完整功能
- 优化算法
- 图形界面接口
- 企业集成功能
- 高性能计算

这个Python重新实现方案保持了DWSIM5的核心架构思想，同时利用了Python生态的优势，提供了现代化的API服务接口。 