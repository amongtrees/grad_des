import json

from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer

class SnifferConsumer(WebsocketConsumer):

    def connect(self):
        self.group_name = 'sniffer_group'
        async_to_sync(self.channel_layer.group_add)(
            self.group_name,
            self.channel_name
        )
        self.accept()
        print("WebSocket connection established")

    def disconnect(self, close_code):
        async_to_sync(self.channel_layer.group_discard)(
            self.group_name,
            self.channel_name
        )
        print("WebSocket disconnected")

    def send_flow_info(self, flow_info):
        """通过 channels_layer 发送数据包信息给前端"""
        try:

            print(f"Sending flow info: {self.channel_name, flow_info}")

            # 获取 channels_layer
            # channel_layer = get_channel_layer()
            #
            # # 将数据发送到特定的 channel
            # channel_layer.send(
            #     self.channel_name,  # 发送到当前 WebSocket 消费者的 channel
            #     {
            #         'type': 'send_flow_info',  # 类型，表示消息内容
            #         'flow_info': flow_info,  # 实际的数据
            #     }
            # )
            self.send(text_data=json.dumps(flow_info))
            print(f"Flow info sent: {flow_info}")
        except Exception as e:
            print(f"Error sending flow info in websocket: {e}")

    def receive(self, text_data):
        """接收来自前端的消息并返回"""
        # print(f"Received message: {text_data}")
        # text_data_json = json.loads(text_data)
        # message = '运维咖啡吧：' + text_data_json.get('message', 'No message')
        #
        # # 将返回的消息发送给前端
        # self.send(text_data=json.dumps({
        #     'message': message
        # }))

    def close_connection(self):
        """手动关闭 WebSocket 连接"""
        # 通过调用 `self.close` 来关闭 WebSocket
        if self.channel_name:  # 确保已经连接
            self.close()
