from django.test import TestCase

# Create your tests here.

from scapy.all import get_if_list  # 导入正确的函数

if __name__ == '__main__':
    # 获取所有可用的网络接口名称列表
    interfaces = get_if_list()
    for iface in interfaces:
        print(iface)