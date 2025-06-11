"""
矩阵操作模块
============

包含矩阵的各种数学运算，如行列式计算、矩阵求逆、LU分解等。
这些功能是从DWSIM.Math/MatrixOps.vb转换而来的。

主要功能:
- 行列式计算
- 矩阵求逆
- LU分解
- 三角矩阵求逆

作者: DWSIM团队 (Python转换版本)
许可证: GNU General Public License v3.0
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings
# from ..solvers import linear_system  # 将在后面创建


class Determinant:
    """
    行列式计算类
    
    提供各种方法计算矩阵的行列式，包括一般矩阵和LU分解矩阵的行列式计算。
    """
    
    @staticmethod
    def matrix_determinant(matrix: Union[np.ndarray, list]) -> float:
        """
        计算一般矩阵的行列式
        
        使用LU分解方法计算矩阵的行列式。对于大型矩阵，这比直接展开更高效。
        
        参数:
            matrix: 输入矩阵，可以是numpy数组或二维列表
            
        返回:
            float: 矩阵的行列式值
            
        示例:
            >>> A = [[1, 2], [3, 4]]
            >>> Determinant.matrix_determinant(A)
            -2.0
        """
        A = np.asarray(matrix, dtype=float)
        
        if A.ndim != 2:
            raise ValueError("输入必须是二维矩阵")
            
        n = A.shape[0]
        if A.shape[1] != n:
            raise ValueError("输入必须是方阵")
        
        if n == 0:
            return 1.0
        elif n == 1:
            return float(A[0, 0])
        elif n == 2:
            return float(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
        else:
            # 使用LU分解计算行列式
            A_copy = A.copy()
            pivots = np.zeros(n, dtype=int)
            
            try:
                # 执行LU分解
                LUDecomposition.lu_decomposition(A_copy, n, n, pivots)
                return Determinant.lu_matrix_determinant(A_copy, pivots, n)
            except Exception as e:
                # 如果LU分解失败，使用numpy的内置函数作为备选
                warnings.warn(f"LU分解失败，使用numpy计算: {e}")
                return float(np.linalg.det(A))
    
    @staticmethod
    def lu_matrix_determinant(lu_matrix: np.ndarray, pivots: np.ndarray, n: int) -> float:
        """
        从LU分解结果计算行列式
        
        根据LU分解的结果和主元置换信息计算原矩阵的行列式。
        
        参数:
            lu_matrix: LU分解后的矩阵
            pivots: 主元置换数组
            n: 矩阵大小
            
        返回:
            float: 行列式值
            
        公式:
            det(A) = det(P) * det(L) * det(U) = (-1)^(置换次数) * 1 * ∏(U_ii)
        """
        if n == 0:
            return 1.0
        
        # 计算上三角矩阵U的对角元素乘积
        det = 1.0
        for i in range(n):
            det *= lu_matrix[i, i]
        
        # 计算置换矩阵P的行列式（-1的置换次数次方）
        perm_sign = 1
        for i in range(n):
            if pivots[i] != i:
                perm_sign = -perm_sign
        
        return float(det * perm_sign)


class Inverse:
    """
    矩阵求逆类
    
    提供各种方法求矩阵的逆，包括一般矩阵求逆和基于LU分解的求逆。
    """
    
    @staticmethod
    def matrix_inverse(matrix: Union[np.ndarray, list]) -> Tuple[np.ndarray, bool]:
        """
        计算一般矩阵的逆矩阵
        
        使用LU分解方法求矩阵的逆。如果矩阵奇异，则返回失败标志。
        
        参数:
            matrix: 输入矩阵，可以是numpy数组或二维列表
            
        返回:
            tuple: (逆矩阵, 是否成功)
            
        示例:
            >>> A = [[2, 1], [1, 1]]
            >>> inv_A, success = Inverse.matrix_inverse(A)
            >>> if success:
            ...     print("逆矩阵计算成功")
        """
        A = np.asarray(matrix, dtype=float)
        
        if A.ndim != 2:
            raise ValueError("输入必须是二维矩阵")
            
        n = A.shape[0]
        if A.shape[1] != n:
            raise ValueError("输入必须是方阵")
        
        if n == 0:
            return np.array([[]]), True
        
        A_copy = A.copy()
        pivots = np.zeros(n, dtype=int)
        
        try:
            # 执行LU分解
            LUDecomposition.lu_decomposition(A_copy, n, n, pivots)
            success = Inverse.lu_matrix_inverse(A_copy, pivots, n)
            return A_copy, success
        except Exception as e:
            warnings.warn(f"LU分解求逆失败: {e}")
            return A, False
    
    @staticmethod
    def lu_matrix_inverse(lu_matrix: np.ndarray, pivots: np.ndarray, n: int) -> bool:
        """
        基于LU分解结果计算矩阵的逆
        
        根据LU分解的结果计算原矩阵的逆矩阵，结果直接存储在lu_matrix中。
        
        参数:
            lu_matrix: LU分解后的矩阵，计算完成后包含逆矩阵
            pivots: 主元置换数组
            n: 矩阵大小
            
        返回:
            bool: 是否成功计算逆矩阵
        """
        if n == 0:
            return True
        
        try:
            # 先计算上三角矩阵的逆
            success = TriangularInverse.triangular_inverse(
                lu_matrix, n, is_upper=True, is_unit_triangular=False
            )
            
            if not success:
                return False
            
            # 处理下三角部分的逆运算
            temp = np.zeros(n)
            
            # 从下到上处理每一列
            for j in range(n-1, -1, -1):
                # 保存当前列的下三角部分
                for i in range(j+1, n):
                    temp[i] = lu_matrix[i, j]
                    lu_matrix[i, j] = 0.0
                
                # 如果不是最后一列，需要更新
                if j < n-1:
                    for i in range(n):
                        sum_val = 0.0
                        for k in range(j+1, n):
                            sum_val += lu_matrix[i, k] * temp[k]
                        lu_matrix[i, j] -= sum_val
            
            # 应用列置换
            for j in range(n-2, -1, -1):
                pivot_col = pivots[j]
                if pivot_col != j:
                    # 交换列j和pivot_col
                    for i in range(n):
                        temp_val = lu_matrix[i, j]
                        lu_matrix[i, j] = lu_matrix[i, pivot_col]
                        lu_matrix[i, pivot_col] = temp_val
            
            return True
            
        except Exception as e:
            warnings.warn(f"LU逆矩阵计算失败: {e}")
            return False


class TriangularInverse:
    """
    三角矩阵求逆类
    
    专门用于求解上三角或下三角矩阵的逆矩阵。
    """
    
    @staticmethod
    def triangular_inverse(matrix: np.ndarray, n: int, 
                          is_upper: bool = True, 
                          is_unit_triangular: bool = False) -> bool:
        """
        计算三角矩阵的逆矩阵
        
        支持四种类型的三角矩阵:
        - 上三角矩阵
        - 单位上三角矩阵（对角线元素为1）
        - 下三角矩阵  
        - 单位下三角矩阵（对角线元素为1）
        
        参数:
            matrix: 输入三角矩阵，结果将直接存储在此矩阵中
            n: 矩阵大小
            is_upper: 是否为上三角矩阵
            is_unit_triangular: 是否为单位三角矩阵（对角线为1）
            
        返回:
            bool: 是否成功计算逆矩阵
        """
        if n == 0:
            return True
        
        temp = np.zeros(n)
        has_diagonal = not is_unit_triangular
        
        try:
            if is_upper:
                return TriangularInverse._upper_triangular_inverse(
                    matrix, n, has_diagonal, temp
                )
            else:
                return TriangularInverse._lower_triangular_inverse(
                    matrix, n, has_diagonal, temp
                )
        except Exception as e:
            warnings.warn(f"三角矩阵求逆失败: {e}")
            return False
    
    @staticmethod
    def _upper_triangular_inverse(matrix: np.ndarray, n: int, 
                                 has_diagonal: bool, temp: np.ndarray) -> bool:
        """计算上三角矩阵的逆"""
        for j in range(n):
            if has_diagonal:
                if matrix[j, j] == 0:
                    return False
                matrix[j, j] = 1.0 / matrix[j, j]
                ajj = -matrix[j, j]
            else:
                ajj = -1.0
            
            if j > 0:
                # 保存第j列的上三角部分
                for i in range(j):
                    temp[i] = matrix[i, j]
                
                # 计算逆矩阵的第j列
                for i in range(j):
                    if i < j-1:
                        sum_val = 0.0
                        for k in range(i+1, j):
                            sum_val += matrix[i, k] * temp[k]
                    else:
                        sum_val = 0.0
                    
                    if has_diagonal:
                        matrix[i, j] = sum_val + matrix[i, i] * temp[i]
                    else:
                        matrix[i, j] = sum_val + temp[i]
                
                # 乘以对角元素的负逆
                for i in range(j):
                    matrix[i, j] *= ajj
        
        return True
    
    @staticmethod
    def _lower_triangular_inverse(matrix: np.ndarray, n: int, 
                                 has_diagonal: bool, temp: np.ndarray) -> bool:
        """计算下三角矩阵的逆"""
        for j in range(n-1, -1, -1):
            if has_diagonal:
                if matrix[j, j] == 0:
                    return False
                matrix[j, j] = 1.0 / matrix[j, j]
                ajj = -matrix[j, j]
            else:
                ajj = -1.0
            
            if j < n-1:
                # 保存第j列的下三角部分
                for i in range(j+1, n):
                    temp[i] = matrix[i, j]
                
                # 计算逆矩阵的第j列
                for i in range(j+1, n):
                    if i > j+1:
                        sum_val = 0.0
                        for k in range(j+1, i):
                            sum_val += matrix[i, k] * temp[k]
                    else:
                        sum_val = 0.0
                    
                    if has_diagonal:
                        matrix[i, j] = sum_val + matrix[i, i] * temp[i]
                    else:
                        matrix[i, j] = sum_val + temp[i]
                
                # 乘以对角元素的负逆
                for i in range(j+1, n):
                    matrix[i, j] *= ajj
        
        return True


class LUDecomposition:
    """
    LU分解类
    
    提供矩阵的LU分解功能，支持部分主元选择。
    """
    
    @staticmethod
    def lu_decomposition(matrix: np.ndarray, m: int, n: int, pivots: np.ndarray) -> bool:
        """
        执行矩阵的LU分解（带部分主元选择）
        
        将矩阵A分解为PA = LU的形式，其中P是置换矩阵，L是单位下三角矩阵，
        U是上三角矩阵。
        
        参数:
            matrix: 输入矩阵，分解结果将直接存储在此矩阵中
            m: 矩阵的行数
            n: 矩阵的列数
            pivots: 存储主元置换信息的数组
            
        返回:
            bool: 是否成功完成LU分解
        """
        if m == 0 or n == 0:
            return True
        
        try:
            # 初始化置换数组
            for i in range(min(m, n)):
                pivots[i] = i
            
            # LU分解的主循环
            for k in range(min(m-1, n)):
                # 寻找主元
                pivot_row = k
                max_val = abs(matrix[k, k])
                
                for i in range(k+1, m):
                    if abs(matrix[i, k]) > max_val:
                        max_val = abs(matrix[i, k])
                        pivot_row = i
                
                # 如果主元为零，矩阵奇异
                if max_val == 0:
                    warnings.warn(f"矩阵在第{k}步出现零主元，可能奇异")
                    return False
                
                # 执行行交换
                if pivot_row != k:
                    pivots[k] = pivot_row
                    for j in range(n):
                        temp = matrix[k, j]
                        matrix[k, j] = matrix[pivot_row, j]
                        matrix[pivot_row, j] = temp
                
                # 计算下三角部分的乘数
                for i in range(k+1, m):
                    matrix[i, k] /= matrix[k, k]
                
                # 更新剩余子矩阵
                for i in range(k+1, m):
                    for j in range(k+1, n):
                        matrix[i, j] -= matrix[i, k] * matrix[k, j]
            
            return True
            
        except Exception as e:
            warnings.warn(f"LU分解过程中出错: {e}")
            return False


class MatrixOperations:
    """
    矩阵操作的高级接口类
    
    提供简化的矩阵操作接口，整合各种矩阵运算功能。
    """
    
    @staticmethod
    def determinant(matrix: Union[np.ndarray, list]) -> float:
        """计算矩阵行列式的便捷方法"""
        return Determinant.matrix_determinant(matrix)
    
    @staticmethod
    def inverse(matrix: Union[np.ndarray, list]) -> Tuple[np.ndarray, bool]:
        """计算矩阵逆的便捷方法"""
        return Inverse.matrix_inverse(matrix)
    
    @staticmethod
    def solve_linear_system(A: Union[np.ndarray, list], 
                           b: Union[np.ndarray, list]) -> Tuple[np.ndarray, bool]:
        """
        求解线性方程组 Ax = b
        
        参数:
            A: 系数矩阵
            b: 右端向量
            
        返回:
            tuple: (解向量, 是否成功)
        """
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        
        if A.shape[0] != len(b):
            raise ValueError("系数矩阵和右端向量维数不匹配")
        
        try:
            x = np.linalg.solve(A, b)
            return x, True
        except np.linalg.LinAlgError:
            return np.zeros_like(b), False
    
    @staticmethod
    def condition_number(matrix: Union[np.ndarray, list]) -> float:
        """
        计算矩阵的条件数
        
        条件数衡量矩阵的病态程度，条件数越大，矩阵越病态。
        
        参数:
            matrix: 输入矩阵
            
        返回:
            float: 条件数
        """
        A = np.asarray(matrix, dtype=float)
        return float(np.linalg.cond(A))
    
    @staticmethod
    def rank(matrix: Union[np.ndarray, list]) -> int:
        """
        计算矩阵的秩
        
        参数:
            matrix: 输入矩阵
            
        返回:
            int: 矩阵的秩
        """
        A = np.asarray(matrix, dtype=float)
        return int(np.linalg.matrix_rank(A))


# 为了保持向后兼容性，提供一些便捷函数
def determinant(matrix: Union[np.ndarray, list]) -> float:
    """计算行列式的便捷函数"""
    return MatrixOperations.determinant(matrix)

def inverse(matrix: Union[np.ndarray, list]) -> Tuple[np.ndarray, bool]:
    """计算逆矩阵的便捷函数"""
    return MatrixOperations.inverse(matrix)

def solve(A: Union[np.ndarray, list], b: Union[np.ndarray, list]) -> Tuple[np.ndarray, bool]:
    """求解线性方程组的便捷函数"""
    return MatrixOperations.solve_linear_system(A, b) 