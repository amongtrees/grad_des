import csv
import os
from collections import defaultdict
from model.classification import Classifier
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from scapy.layers.tls.record import TLS, TLSApplicationData
from scapy.sessions import TCPSession
from meter.features.context.packet_direction import PacketDirection
from meter.features.context.packet_flow_key import get_packet_flow_key
from meter.flow import Flow
from meter.time_series.processor import Processor

EXPIRED_UPDATE = 40


class FlowSession(TCPSession):
    """Creates a list of network flows."""

    def __init__(self, *args, **kwargs):
        self.flows = {}
        self.csv_line = 0

        if self.output_mode == 'flow':
            output = open(self.output_file, 'w')
            self.csv_writer = csv.writer(output)

        self.packets_count = 0

        self.clumped_flows_per_label = defaultdict(list)

        self.classifier_1 = Classifier(r"E:\Graduate_Design\project\model\is_non_model.pth", r"E:\Graduate_Design\project\CSVs\Total_CSVs\l1-doh.csv")

        self.classifier_2 = Classifier(r"E:\Graduate_Design\project\model\ben_mal_model.pth", r"E:\Graduate_Design\project\CSVs\Total_CSVs\l2-benign.csv")

        super(FlowSession, self).__init__(None, True, *args, **kwargs)

    def toPacketList(self):
        # Sniffer finished all the packets it needed to sniff.
        # It is not a good place for this, we need to somehow define a finish signal for AsyncSniffer
        self.garbage_collect(None)
        return super(FlowSession, self).toPacketList()

    def on_packet_received(self, packet):
        count = 0
        direction = PacketDirection.FORWARD

        if self.output_mode != 'flow':
            if TLS not in packet:
                return

            if TLSApplicationData not in packet:
                return

            if len(packet[TLSApplicationData]) < 40:
                # PING frame (len = 34) or other useless frames
                return

        self.packets_count += 1

        # Creates a key variable to check
        packet_flow_key = get_packet_flow_key(packet, direction)
        flow = self.flows.get((packet_flow_key, count))

        # If there is no forward flow with a count of 0
        if flow is None:
            # There might be one of it in reverse
            direction = PacketDirection.REVERSE
            packet_flow_key = get_packet_flow_key(packet, direction)
            flow = self.flows.get((packet_flow_key, count))

            if flow is None:
                # If no flow exists create a new flow
                direction = PacketDirection.FORWARD
                flow = Flow(packet, direction)
                packet_flow_key = get_packet_flow_key(packet, direction)
                self.flows[(packet_flow_key, count)] = flow

            elif (packet.time - flow.latest_timestamp) > EXPIRED_UPDATE:
                # If the packet exists in the flow but the packet is sent
                # after too much of a delay than it is a part of a new flow.
                expired = EXPIRED_UPDATE
                while (packet.time - flow.latest_timestamp) > expired:
                    count += 1
                    expired += EXPIRED_UPDATE
                    flow = self.flows.get((packet_flow_key, count))

                    if flow is None:
                        flow = Flow(packet, direction)
                        self.flows[(packet_flow_key, count)] = flow
                        break

        elif (packet.time - flow.latest_timestamp) > EXPIRED_UPDATE:
            expired = EXPIRED_UPDATE
            while (packet.time - flow.latest_timestamp) > expired:

                count += 1
                expired += EXPIRED_UPDATE
                flow = self.flows.get((packet_flow_key, count))

                if flow is None:
                    flow = Flow(packet, direction)
                    self.flows[(packet_flow_key, count)] = flow
                    break

        flow.add_packet(packet, direction)

        if self.packets_count % 10000 == 0 or (flow.duration > 120 and self.output_mode == 'flow'):
            print('Packet count: {}'.format(self.packets_count))
            self.garbage_collect(packet.time)

    def get_flows(self) -> list:
        return self.flows.values()

    def garbage_collect(self, latest_time) -> None:
        # TODO: Garbage Collection / Feature Extraction should have a separate thread
        print('Garbage Collection Began. Flows = {}'.format(len(self.flows)))
        keys = list(self.flows.keys())

        for k in keys:
            flow = self.flows.get(k)
            flow_info = {}
            if self.output_mode == 'flow':
                if latest_time is None or latest_time - flow.latest_timestamp > EXPIRED_UPDATE or flow.duration > 90:
                    data = flow.get_data()
                    features = list(data.values())[5:-1]

                    flow_info.update({'src_ip': data['SourceIP']})
                    flow_info.update({'dst_ip': data['DestinationIP']})
                    flow_info.update({'src_port': data['SourcePort']})
                    flow_info.update({'dst_port': data['DestinationPort']})
                    flow_info.update({'time_stamp': data['TimeStamp']})
                    if self.classifier_1.classify(features):
                        flow_info.update({'is_doh': True})
                        if self.classifier_2.classify(features):
                            flow_info.update({'is_benign': True})
                        else:
                            flow_info.update({'is_benign': False})
                    else:
                        flow_info.update({'is_doh': False})
                    # async_to_sync(self.websocket_consumer.send_flow_info(flow_info))
                    channel_layer = get_channel_layer()
                    # print(f"Channel layer: {channel_layer}")  # 确认 channel_layer 是否获取成功

                    try:
                        async_to_sync(channel_layer.group_send)(
                            'sniffer_group',  # 组名
                            {
                                'type': 'send_flow_info',  # 消费者方法名
                                'flow_info': flow_info,  # 数据包信息
                                'cur_mode': self.input_mode,
                            }
                        )
                        print(f"Sent flow info: {flow_info}")  # 确认消息是否成功发送
                    except Exception as e:
                        print(f"Error sending flow info: {e}")
                    if self.csv_line == 0:
                        self.csv_writer.writerow(data.keys())
                    self.csv_writer.writerow(data.values())
                    self.csv_line += 1
                    del self.flows[k]
            else:
                if latest_time is None or latest_time - flow.latest_timestamp > EXPIRED_UPDATE:
                    output_dir = os.path.join(self.output_file, 'doh' if flow.is_doh() else 'ndoh')
                    os.makedirs(output_dir, exist_ok=True)
                    proc = Processor(flow)
                    flow_clumps = proc.create_flow_clumps_container()
                    flow_clumps.to_json_file(output_dir)
                    del self.flows[k]

        print('Garbage Collection Finished. Flows = {}'.format(len(self.flows)))
