# 创建独立线程类
import threading
from threading import Thread
import time

from scapy.sessions import TCPSession

class FlowProcessorThread(Thread):
    def __init__(self, flow_session, interval=5):
        super().__init__()
        self.flow_session = flow_session
        self.interval = interval  # 检查间隔(秒)
        self._running = True

    def run(self):
        while self._running:
            time.sleep(self.interval)
            self.flow_session.garbage_collect(time.time())

    def stop(self):
        self._running = False

# 修改FlowSession类
class FlowSession(TCPSession):
    def __init__(self, *args, **kwargs):
        # ... 原有初始化代码 ...
        self.processor_thread = FlowProcessorThread(self)
        self.processor_thread.start()

    def __del__(self):
        if hasattr(self, 'processor_thread'):
            self.processor_thread.stop()
            self.processor_thread.join()

    def garbage_collect(self, latest_time) -> None:
        # 加锁防止多线程竞争
        with threading.Lock():
            keys = list(self.flows.keys())
            flows_to_process = []

            # 快速筛选需要处理的流
            for k in keys:
                flow = self.flows.get(k)
                if (latest_time is None or
                        latest_time - flow.latest_timestamp > EXPIRED_UPDATE or
                        flow.duration > 90):
                    flows_to_process.append((k, flow))

            # 批量处理流
            for k, flow in flows_to_process:
                self._process_flow(flow)
                del self.flows[k]

    def _process_flow(self, flow):
        """处理单个流的通用方法"""
        flow_info = {
            'src_ip': flow.src_ip,
            'dst_ip': flow.dst_ip,
            # ... 其他字段 ...
        }

        if self.output_mode == 'flow':
            data = flow.get_data()
            features = list(data.values())[5:-1]

            # 分类处理
            flow_info['is_doh'] = self.classifier_1.classify(features)
            if flow_info['is_doh']:
                flow_info['is_benign'] = self.classifier_2.classify(features)

            # 异步发送结果
            self._send_flow_info(flow_info)

            # 写入CSV
            if self.csv_line == 0:
                self.csv_writer.writerow(data.keys())
            self.csv_writer.writerow(data.values())
            self.csv_line += 1
        else:
            # 离线模式处理
            output_dir = os.path.join(self.output_file, 'doh' if flow.is_doh() else 'ndoh')
            os.makedirs(output_dir, exist_ok=True)
            proc = Processor(flow)
            flow_clumps = proc.create_flow_clumps_container()
            flow_clumps.to_json_file(output_dir)

    def _send_flow_info(self, flow_info):
        try:
            channel_layer = get_channel_layer()
            async_to_sync(channel_layer.group_send)(
                'sniffer_group',
                {
                    'type': 'send_flow_info',
                    'flow_info': flow_info,
                    'cur_mode': self.input_mode,
                }
            )
        except Exception as e:
            logging.error(f"Error sending flow info: {e}")