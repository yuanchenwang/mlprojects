import logging 
import os 
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
                    
                    
                                       
)

if __name__=="__main__":
    logging.info("Logging has started")


"""
import logging: 导入Python的日志模块，用于记录日志。
import os: 导入操作系统模块，用于处理文件和目录。
from datetime import datetime: 从datetime模块导入datetime类，用于处理日期和时间。
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log":

创建一个日志文件名，格式为"月_日_年_时_分_秒.log"。
例如：09_03_2024_14_30_45.log


logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE):

os.getcwd() 获取当前工作目录。
将当前目录、"logs"文件夹和LOG_FILE拼接成一个完整的路径。
例如：/home/user/project/logs/09_03_2024_14_30_45.log


os.makedirs(logs_path,exist_ok=True):

创建logs_path指定的目录。
exist_ok=True 表示如果目录已存在，不会报错。


LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE):

将日志文件路径和文件名组合成完整的文件路径。



8-14. logging.basicConfig(...):

配置日志系统：

filename=LOG_FILE_PATH: 指定日志文件的位置。
format="...": 设置日志消息的格式。
level=logging.INFO: 设置日志级别为INFO。



15-16. if __name__=="__main__": 和 logging.info("Logging has started"):

如果这个脚本是直接运行的（不是被导入的），则执行logging.info()。
在日志中记录一条"Logging has started"的信息。

现在，让我们通过一个例子来说明这段代码的执行过程：
假设你在2024年9月3日14:30:45运行这个脚本，以下是会发生的事情：

创建日志文件名：09_03_2024_14_30_45.log
假设你的当前工作目录是 /home/user/project，那么完整的日志路径将是：
/home/user/project/logs/09_03_2024_14_30_45.log
如果 /home/user/project/logs 目录不存在，它会被创建。
日志系统被配置，准备写入上面指定的文件。
最后，一条日志信息被写入文件。
"""