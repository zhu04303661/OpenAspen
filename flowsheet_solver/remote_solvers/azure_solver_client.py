"""
Azure云计算求解器客户端
=====================

实现通过Microsoft Azure Service Bus进行远程流程图计算的客户端。
支持数据压缩、分片传输、超时处理等功能。

对应原VB.NET版本中的AzureSolverClient类。
"""

import time
import gzip
import uuid
import logging
from typing import Optional, Any, List, Dict
from dataclasses import dataclass
from io import BytesIO

try:
    from azure.servicebus import ServiceBusClient, ServiceBusMessage, ServiceBusReceiver
    from azure.servicebus.exceptions import ServiceBusError
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

try:
    from ..solver_exceptions import NetworkException, TimeoutException, DataException
except ImportError:
    from flowsheet_solver.solver_exceptions import NetworkException, TimeoutException, DataException


@dataclass
class AzureConnectionSettings:
    """
    Azure连接设置
    """
    connection_string: str
    server_queue_name: str = "DWSIMserver"
    client_queue_name: str = "DWSIMclient"
    timeout_seconds: float = 300
    max_message_size: int = 220 * 1024  # 220KB


class AzureSolverClient:
    """
    Azure Service Bus 求解器客户端
    
    实现通过Azure Service Bus队列与远程求解服务器通信的功能。
    支持流程图数据的压缩传输、分片处理和结果接收。
    """
    
    def __init__(self, connection_string: str, queue_name: str = "DWSIMserver", timeout_seconds: float = 300):
        """
        初始化Azure求解器客户端
        
        Args:
            connection_string: Azure Service Bus连接字符串
            queue_name: 队列名称
            timeout_seconds: 超时时间（秒）
        """
        if not AZURE_AVAILABLE:
            raise ImportError("Azure Service Bus依赖不可用，请安装azure-servicebus包")
        
        self.settings = AzureConnectionSettings(
            connection_string=connection_string,
            server_queue_name=queue_name,
            timeout_seconds=timeout_seconds
        )
        self.logger = logging.getLogger(__name__)
        
        self.service_bus_client: Optional[ServiceBusClient] = None
        self.server_sender = None
        self.client_receiver = None
        self.is_connected = False  # 添加测试期望的属性
    
    def connect(self):
        """建立连接（公共接口）"""
        self._connect()
        
    def solve_flowsheet(self, flowsheet: Any) -> List[Exception]:
        """
        求解流程图
        
        将流程图数据发送到Azure Service Bus，等待远程服务器计算完成并接收结果。
        
        Args:
            flowsheet: 流程图对象
            
        Returns:
            List[Exception]: 异常列表
        """
        self.logger.info("开始Azure云计算求解")
        
        try:
            # 建立连接
            self._connect()
            
            # 检查服务器可用性
            request_id = self._check_server_availability()
            
            # 发送流程图数据
            self._send_flowsheet_data(flowsheet, request_id)
            
            # 等待并接收结果
            self._receive_and_process_results(flowsheet, request_id)
            
            return []  # 返回空异常列表表示成功
            
        except Exception as e:
            self.logger.error(f"Azure求解失败: {e}")
            raise NetworkException(f"Azure云计算失败: {str(e)}")
        
        finally:
            # 清理连接
            self._disconnect()
    
    def _connect(self) -> None:
        """
        建立Azure Service Bus连接
        """
        try:
            self.logger.info("连接到Azure Service Bus")
            
            # 创建Service Bus客户端
            self.service_bus_client = ServiceBusClient.from_connection_string(
                self.settings.connection_string
            )
            
            # 创建发送器和接收器
            self.server_sender = self.service_bus_client.get_queue_sender(
                queue_name=self.settings.server_queue_name
            )
            self.client_receiver = self.service_bus_client.get_queue_receiver(
                queue_name=self.settings.client_queue_name
            )
            
            # 清空客户端队列中的残留消息
            self._clear_queue_messages()
            
            self.logger.info("Azure Service Bus连接成功")
            
            self.is_connected = True
            
        except ServiceBusError as e:
            raise NetworkException(f"Azure Service Bus连接失败: {str(e)}")
    
    def _disconnect(self) -> None:
        """
        断开Azure Service Bus连接
        """
        try:
            if self.server_sender:
                self.server_sender.close()
            if self.client_receiver:
                self.client_receiver.close()
            if self.service_bus_client:
                self.service_bus_client.close()
                
            self.logger.info("Azure Service Bus连接已断开")
            
            self.is_connected = False
            
        except Exception as e:
            self.logger.warning(f"断开Azure连接时出错: {e}")
    
    def _clear_queue_messages(self) -> None:
        """
        清空队列中的残留消息
        """
        try:
            # 清空客户端队列
            messages = self.client_receiver.receive_messages(
                max_message_count=100,
                max_wait_time=1
            )
            for message in messages:
                self.client_receiver.complete_message(message)
                
            self.logger.debug("已清空队列中的残留消息")
            
        except Exception as e:
            self.logger.warning(f"清空队列消息时出错: {e}")
    
    def _check_server_availability(self) -> str:
        """
        检查服务器可用性
        
        Returns:
            str: 请求ID
        """
        request_id = str(uuid.uuid4())
        
        try:
            # 发送连接检查消息
            message = ServiceBusMessage("")
            message.application_properties = {
                "requestID": request_id,
                "type": "connectioncheck",
                "origin": "client"
            }
            
            self.server_sender.send_messages(message)
            self.logger.info("已发送服务器可用性检查请求")
            
            # 等待服务器响应
            start_time = time.time()
            timeout = 20  # 20秒超时
            
            while time.time() - start_time < timeout:
                messages = self.client_receiver.receive_messages(
                    max_message_count=1,
                    max_wait_time=1
                )
                
                for message in messages:
                    props = message.application_properties or {}
                    
                    if (props.get("requestID") == request_id and
                        props.get("origin") == "server" and
                        props.get("type") == "connectioncheck"):
                        
                        self.client_receiver.complete_message(message)
                        self.logger.info("服务器可用性检查通过")
                        return request_id
                    
                    self.client_receiver.complete_message(message)
            
            raise TimeoutException("服务器可用性检查超时")
            
        except ServiceBusError as e:
            raise NetworkException(f"服务器可用性检查失败: {str(e)}")
    
    def _send_flowsheet_data(self, flowsheet: Any, request_id: str) -> None:
        """
        发送流程图数据
        
        Args:
            flowsheet: 流程图对象
            request_id: 请求ID
        """
        try:
            # 序列化流程图数据（这里需要根据具体的flowsheet对象实现）
            xml_data = self._serialize_flowsheet(flowsheet)
            
            # 压缩数据
            compressed_data = self._compress_data(xml_data)
            
            self.logger.info(f"原始数据大小: {len(xml_data)} 字节")
            self.logger.info(f"压缩后数据大小: {len(compressed_data)} 字节")
            
            # 检查是否需要分片传输
            if len(compressed_data) <= self.settings.max_message_size:
                # 单条消息发送
                self._send_single_message(compressed_data, request_id)
            else:
                # 分片发送
                self._send_multipart_message(compressed_data, request_id)
                
        except Exception as e:
            raise DataException(f"发送流程图数据失败: {str(e)}", len(compressed_data))
    
    def _serialize_flowsheet(self, flowsheet: Any) -> bytes:
        """
        序列化流程图对象
        
        Args:
            flowsheet: 流程图对象
            
        Returns:
            bytes: 序列化后的XML数据
        """
        # 这里需要根据具体的flowsheet对象实现序列化
        # 示例实现：
        if hasattr(flowsheet, 'save_to_xml'):
            xml_data = flowsheet.save_to_xml()
            if isinstance(xml_data, str):
                return xml_data.encode('utf-8')
            return xml_data
        else:
            # 如果没有save_to_xml方法，返回空数据
            self.logger.warning("流程图对象没有save_to_xml方法，返回空数据")
            return b"<empty/>"
    
    def _compress_data(self, data: bytes) -> bytes:
        """
        压缩数据
        
        Args:
            data: 原始数据
            
        Returns:
            bytes: 压缩后的数据
        """
        compressed_stream = BytesIO()
        with gzip.GzipFile(fileobj=compressed_stream, mode='wb') as gz:
            gz.write(data)
        return compressed_stream.getvalue()
    
    def _send_single_message(self, data: bytes, request_id: str) -> None:
        """
        发送单条消息
        
        Args:
            data: 数据
            request_id: 请求ID
        """
        message = ServiceBusMessage(data)
        message.application_properties = {
            "multipart": False,
            "requestID": request_id,
            "origin": "client",
            "type": "data"
        }
        
        self.server_sender.send_messages(message)
        self.logger.info(f"已发送单条消息，大小: {len(data)} 字节")
    
    def _send_multipart_message(self, data: bytes, request_id: str) -> None:
        """
        发送分片消息
        
        Args:
            data: 数据
            request_id: 请求ID
        """
        chunk_size = self.settings.max_message_size
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        total_parts = len(chunks)
        
        self.logger.info(f"开始分片传输，总片数: {total_parts}")
        
        for i, chunk in enumerate(chunks):
            message = ServiceBusMessage(chunk)
            message.application_properties = {
                "multipart": True,
                "partnumber": i + 1,
                "totalparts": total_parts,
                "requestID": request_id,
                "origin": "client",
                "type": "data"
            }
            
            self.server_sender.send_messages(message)
            self.logger.debug(f"已发送分片 {i + 1}/{total_parts}，大小: {len(chunk)} 字节")
    
    def _receive_and_process_results(self, flowsheet: Any, request_id: str) -> None:
        """
        接收并处理计算结果
        
        Args:
            flowsheet: 流程图对象
            request_id: 请求ID
        """
        self.logger.info("等待计算结果")
        
        start_time = time.time()
        received_parts: Dict[int, bytes] = {}
        total_parts = 0
        is_multipart = False
        
        while time.time() - start_time < self.settings.timeout_seconds:
            messages = self.client_receiver.receive_messages(
                max_message_count=1,
                max_wait_time=1
            )
            
            for message in messages:
                props = message.application_properties or {}
                
                if (props.get("requestID") == request_id and
                    props.get("origin") == "server" and
                    props.get("type") == "data"):
                    
                    # 处理数据消息
                    if props.get("multipart", False):
                        # 分片消息
                        part_number = props.get("partnumber", 1)
                        total_parts = props.get("totalparts", 1)
                        is_multipart = True
                        
                        received_parts[part_number] = bytes(message)
                        self.logger.debug(f"收到分片 {part_number}/{total_parts}")
                        
                        # 检查是否收到所有分片
                        if len(received_parts) == total_parts:
                            # 重组数据
                            complete_data = b""
                            for i in range(1, total_parts + 1):
                                complete_data += received_parts[i]
                            
                            self._process_result_data(flowsheet, complete_data)
                            self.client_receiver.complete_message(message)
                            return
                    else:
                        # 单条消息
                        self._process_result_data(flowsheet, bytes(message))
                        self.client_receiver.complete_message(message)
                        return
                
                self.client_receiver.complete_message(message)
        
        # 超时
        raise TimeoutException(f"等待计算结果超时: {self.settings.timeout_seconds}秒")
    
    def _process_result_data(self, flowsheet: Any, compressed_data: bytes) -> None:
        """
        处理结果数据
        
        Args:
            flowsheet: 流程图对象
            compressed_data: 压缩的结果数据
        """
        try:
            # 解压缩数据
            decompressed_data = self._decompress_data(compressed_data)
            
            self.logger.info(f"收到结果数据，解压后大小: {len(decompressed_data)} 字节")
            
            # 更新流程图数据（这里需要根据具体的flowsheet对象实现）
            self._update_flowsheet_data(flowsheet, decompressed_data)
            
            self.logger.info("流程图数据更新完成")
            
        except Exception as e:
            raise DataException(f"处理结果数据失败: {str(e)}")
    
    def _decompress_data(self, compressed_data: bytes) -> bytes:
        """
        解压缩数据
        
        Args:
            compressed_data: 压缩数据
            
        Returns:
            bytes: 解压后的数据
        """
        compressed_stream = BytesIO(compressed_data)
        with gzip.GzipFile(fileobj=compressed_stream, mode='rb') as gz:
            return gz.read()
    
    def _update_flowsheet_data(self, flowsheet: Any, xml_data: bytes) -> None:
        """
        更新流程图数据
        
        Args:
            flowsheet: 流程图对象
            xml_data: XML数据
        """
        # 这里需要根据具体的flowsheet对象实现数据更新
        # 示例实现：
        if hasattr(flowsheet, 'update_process_data'):
            # 如果xml_data是bytes，需要转换为字符串或XML对象
            xml_str = xml_data.decode('utf-8') if isinstance(xml_data, bytes) else xml_data
            flowsheet.update_process_data(xml_str)
        else:
            self.logger.warning("流程图对象没有update_process_data方法")
    
    @staticmethod
    def split_data(data: bytes, chunk_size: int) -> List[bytes]:
        """
        分割数据为指定大小的片段
        
        Args:
            data: 原始数据
            chunk_size: 片段大小
            
        Returns:
            List[bytes]: 数据片段列表
        """
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)] 