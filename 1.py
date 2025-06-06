import json
import re
from datetime import datetime, timezone, timedelta, date
from zoneinfo import ZoneInfo
import calendar
import pandas as pd
import zarr
import numpy as np
import pygrib
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from threading import Lock
import sys

# 配置日志，输出到终端和文件
current_date = date.today().strftime("%Y_%m_%d")
log_format = "%(asctime)s - %(levelname)s - %(message)s"
log_file = f"/home/admin123/project/code/new_ec/log/ec/ec_{current_date}.log"

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(log_format))

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter(log_format))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class ThreadSafeDict:
    def __init__(self, initial_dict=None):
        # 如果没有初始字典，则初始化为空字典
        self.data = initial_dict if initial_dict is not None else {}
        self.lock = Lock()

    def update(self, key, value):
        """线程安全地更新某个键对应的整个值"""
        with self.lock:
            if key in self.data:
                self.data[key] = value
            else:
                raise KeyError(f"Key '{key}' not found")

    def append_field(self, key, field, value):
        """
        线程安全地向键对应字典的指定字段（必须是列表）中追加数据。
        如果键不存在，则先用默认结构初始化该键。
        """
        with self.lock:
            if key not in self.data:
                # 根据需要初始化默认结构
                self.data[key] = {"hpa_surf": "", "forecast": [], "data": []}
            # 检查指定字段是否存在且为列表
            if field in self.data[key] and isinstance(self.data[key][field], list):
                self.data[key][field].append(value)
            else:
                raise KeyError(
                    f"Field '{field}' not found or is not a list in key '{key}'"
                )

    def update_field(self, key, field, value):
        """更新某个键对应字典的指定字段（非列表）"""
        with self.lock:
            if key not in self.data:
                # 初始化默认结构
                self.data[key] = {"hpa_surf": "", "forecast": [], "data": []}
            self.data[key][field] = value

    def get(self, key):
        """获取键对应的数据"""
        with self.lock:
            return self.data.get(key)

    def get_all(self):
        """返回一个副本"""
        with self.lock:
            return dict(self.data)


def relevel(name):
    # 1. 删除前缀 "level"/"levels" 及其后面的空白
    name = re.sub(r"^(?i:levels?)\s+", "", name.strip())
    # 2. 删除末尾多余的破折号或空白
    name = re.sub(r"\s*-\s*$", "", name)

    # 3. 按空白分词
    tokens = name.split()

    def format_number(token):
        """如果 token 是数值且小数部分接近 0，则转换为整数形式"""
        try:
            num = float(token)
            if abs(num - round(num)) < 1e-8:
                return str(int(round(num)))
            else:
                return token
        except ValueError:
            return token

    new_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # 如果 token 本身是一个区间（如 "100000.00000000001-50000.00000000001" 或 "0-1"）
        range_match = re.match(r"^(-?\d+(?:\.\d+)?)-(-?\d+(?:\.\d+)?)$", token)
        if range_match:
            left = format_number(range_match.group(1))
            right = format_number(range_match.group(2))
            new_tokens.append(left + "-" + right)
            i += 1
            continue

        # 如果 token 是纯数字
        try:
            float(token)
            is_number = True
        except ValueError:
            is_number = False

        # 如果当前 token 为数字，且下一个 token 也是数字，则视作一个区间（例如 "0 10"）
        if is_number and i + 1 < len(tokens):
            try:
                float(tokens[i + 1])
                new_tokens.append(
                    format_number(token) + "-" + format_number(tokens[i + 1])
                )
                i += 2
                continue
            except ValueError:
                new_tokens.append(format_number(token))
                i += 1
                continue
        else:
            new_tokens.append(token)
            i += 1

    # 4. 将所有 token 用下划线连接
    result = "_".join(new_tokens).replace("_", "", 1)
    # 5. 删除数字和字母之间多余的下划线：
    #    若下划线前是数字，后是字母，则直接连接
    # result = re.sub(r'(\d)_+([a-zA-Z])', r'\1\2', result)
    # result = re.sub(r'([a-zA-Z])_+(\d)', r'\1\2', result)
    return result


def standardize_name(name):
    # 2. 替换括号中的内容，并用 _ 连接
    name = re.sub(
        r"\s*\(([^)]+)\)", r"_\1", name
    )  # e.g. "Vorticity (relative)" -> "vorticity_relative"
    # 3. 替换空格为 _
    name = re.sub(
        r"\s+", "_", name
    )  # e.g. "Surface net long-wave radiation" -> "surface_net_long-wave_radiation"
    # 4. 处理单位中的 `/`，转为 `_` (e.g. "m/s" -> "m_s")
    name = name.replace("/", "_")

    return name


def check_with_in(string):
    substrings = ["accum", "avg", "max", "min"]
    for substring in substrings:
        if substring in string:
            return True
    return False


def process_file_worker(file, file_path, crop_params):
    try:
        start = time.time()
        all_data = []
        grbs = pygrib.open(file_path)

        for grb in grbs:
            name = grb.name
            data_fields = str(grb).split(":", maxsplit=7)
            layer = data_fields[4].strip()
            hpa_surf = "InhPa" if layer[-2:] == "Pa" else "surf"
            # name = data_fields[1].strip().replace(' ', '_')
            level_value = data_fields[5].strip()
            # group_name = f"{layer}_{name}_{level_value}"
            # group_name = re.sub(r'[^a-zA-Z0-9_]', '', group_name.replace("level", ''))

            name = data_fields[1].strip()
            new_name = standardize_name(name)
            new_level = relevel(level_value)
            group_name = f"{layer}_{new_name}_{new_level}"

            type_ = data_fields[6]
            if check_with_in(type_):
                step = int(grb["endStep"])
                forecast_time = int((grb.validDate + timedelta(hours=step)).timestamp())
            else:
                forecast_time = int(grb.validDate.timestamp())

            start_row, end_row, start_col, end_col = crop_params
            data = grb.values[start_row:end_row, start_col:end_col]
            all_data.append([group_name, hpa_surf, forecast_time, data.tolist()])
        grbs.close()

        return all_data
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None


class WeatherDataProcessor:
    def __init__(self, root_dir, target_dir, date, max_threads=30):
        self.crop_params = (146, 348, 1014, 1260)
        self.shape1 = int(self.crop_params[1] - self.crop_params[0])
        self.shape2 = int(self.crop_params[3] - self.crop_params[2])
        self.count = 90
        self.root_dir = root_dir
        self.target_dir = target_dir
        # self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        # self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.max_threads = max_threads  # 最大线程数
        self.overall_start_time = time.time()  # 总开始时间
        self.latitudes = None
        self.longitudes = None
        self.grb_num = 0
        # 初始字典为空
        self.all_data = ThreadSafeDict()
        self.start_stamp = 0
        self.hour = "00"
        self.year_month = "202504"
        self.start_times_load_file_path = (
            "/home/admin123/project/code/new_ec/file/start_times.json"
        )
        self.start_times_list = []
        self.parent_folder_name = ""

        self.target_date = date

    def extract_lat_lon(self, sample_file):
        """从样本文件中提取经纬度数据"""
        try:
            start_row, end_row, start_col, end_col = self.crop_params
            grbs = pygrib.open(sample_file)
            self.grb_num = len(grbs)
            grb = grbs.message(1)  # 获取第一条记录
            latitudes, longitudes = grb.latlons()
            self.latitudes = latitudes[:, 0][start_row:end_row].tolist()
            self.longitudes = longitudes[0, :][start_col:end_col].tolist()
            grbs.close()
        except Exception as e:
            logging.error(f"Error extracting lat/lon: {e}")

    # def time_catalog(self, input_string):
    #     """筛选日期数据（以文件夹名称的前8位作为日期）"""
    #     try:
    #         date_string = input_string[:8]
    #         date_obj = datetime.strptime(date_string, "%Y%m%d")
    #         return self.start_date <= date_obj <= self.end_date
    #     except ValueError:
    #         return False

    def thread_update(self, ts_dict, key, hpa_surf, forecast_time, data):
        # 更新 path 和 start，如果需要的话（例如只更新一次，可以在第一次写入后不重复写）
        ts_dict.update_field(key, "hpa_surf", hpa_surf)
        # 对 forecast 和 data 字段进行追加操作
        ts_dict.append_field(key, "forecast", forecast_time)
        ts_dict.append_field(key, "data", data)

    def process_folder(self, root):
        files = os.listdir(root)
        files = [file for file in files if not file.lower().endswith(".txt")]
        file_paths = [os.path.join(root, file) for file in files]
        results = []
        with ProcessPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {
                executor.submit(process_file_worker, file, fp, self.crop_params): file
                for file, fp in zip(files, file_paths)
            }
            total = len(files)  # 文件总数
            processed = 0  # 已处理文件计数器
            step = max(1, total // 100)  # 日志记录间隔，目标大约 100 条日志
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.extend(result)
                processed += 1
                # 当 processed 是 step 的倍数或处理完成时记录日志
                if processed % step == 0 or processed == total:
                    logging.info(
                        f"Processed {processed}/{total} ({(processed / total * 100):.2f}%)"
                    )

        # 合并结果后，在主进程中更新全局数据
        for item in results:
            self.thread_update(self.all_data, item[0], item[1], item[2], item[3])
        # 单独处理用角风速计算风速
        self.get_wind()
        self.write_all_zarr()

        # 计算辐照读相关数据 并保存(3h累计晴空数据，15min瞬时辐照度)

    def get_wind1(self):
        names = {
            "heightAboveGround_100_metre_wind_component_100m": [
                "heightAboveGround_100_metre_U_wind_component_100m",
                "heightAboveGround_100_metre_V_wind_component_100m",
            ],
            "heightAboveGround_10_metre_wind_component_10m": [
                "heightAboveGround_10_metre_U_wind_component_10m",
                "heightAboveGround_10_metre_V_wind_component_10m",
            ],
        }
        all_data_dict = self.all_data.get_all()
        for key, names in names.items():
            u_wind = all_data_dict[names[0]]
            v_wind = all_data_dict[names[1]]
            hpa_surf = u_wind["hpa_surf"]
            forcast = u_wind["forecast"][:90]
            u_wind_value = np.array(u_wind["data"])
            v_wind_value = np.array(v_wind["data"])
            wind_speed = np.sqrt(u_wind_value**2 + v_wind_value**2)
            for i in range(len(forcast)):
                self.thread_update(
                    self.all_data, key, hpa_surf, forcast[i], wind_speed[i].tolist()
                )

    def get_wind(self):
        names = {
            "heightAboveGround_100_metre_wind_component_100m": [
                "heightAboveGround_100_metre_U_wind_component_100m",
                "heightAboveGround_100_metre_V_wind_component_100m",
                "heightAboveGround_100_metre_wind_direction_component_100m",
            ],
            "heightAboveGround_10_metre_wind_component_10m": [
                "heightAboveGround_10_metre_U_wind_component_10m",
                "heightAboveGround_10_metre_V_wind_component_10m",
                "heightAboveGround_10_metre_wind_direction_component_10m",
            ],
        }
        all_data_dict = self.all_data.get_all()
        for key, names in names.items():
            u_wind = all_data_dict[names[0]]
            v_wind = all_data_dict[names[1]]
            hpa_surf = u_wind["hpa_surf"]
            forcast = u_wind["forecast"][:90]
            u_wind_value = np.array(u_wind["data"])
            v_wind_value = np.array(v_wind["data"])
            wind_speed = np.sqrt(u_wind_value**2 + v_wind_value**2)

            # 计算风向
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = v_wind_value / u_wind_value
            angle = np.degrees(np.arctan(ratio))

            # 处理U=0的特殊情况
            angle = np.where(
                u_wind_value == 0,
                np.sign(v_wind_value) * 90,  # U=0时根据V的符号设置90或-90
                angle,
            )

            # 根据条件分支计算风向
            wind_direction = np.where(
                u_wind_value < 0,
                angle + 180,  # 条件2: U<0时+180度
                np.where(
                    v_wind_value < 0,
                    angle + 360,  # 条件3: U>=0且V<0时+360度
                    angle,  # 条件1: U>=0且V>=0时直接使用
                ),
            )
            wind_direction = wind_direction % 360  # 归一化到0-360度

            # 处理U=0且V=0的情况（风速为0时风向设为0）
            wind_direction = np.where(
                (u_wind_value == 0) & (v_wind_value == 0), 0, wind_direction
            )

            # 保存风速和风向
            for i in range(len(forcast)):
                # 更新风速到原key
                self.thread_update(
                    self.all_data, key, hpa_surf, forcast[i], wind_speed[i].tolist()
                )
                # 新增风向数据到新key（在原key后添加'_direction'）
                self.thread_update(
                    self.all_data,
                    names[2],
                    hpa_surf,
                    forcast[i],
                    wind_direction[i].tolist(),
                )

    def process_files(self):
        """遍历所有文件夹和文件，使用多线程处理多天起报数据"""
        # 先提取经纬度（单线程）
        sample_file = (
            "/home/admin123/EC_Atmos_01/Data/2025020100/20250201000000-3h-oper-fc.grib2"
        )
        self.extract_lat_lon(sample_file)

        for root, dirs, files in os.walk(root_dir):
            # sorted_dirs = sorted(dirs, key=lambda d: os.path.getctime(os.path.join(root, d)))
            dirs = sorted(dirs)
            for dir_name in dirs:
                self.start_times_list = self.load_records()
                # if dir_name in self.start_times_list:
                #     continue
                # if not self.time_catalog(dir_name):
                #     continue  # 筛选符合时间要求的文件夹
                folder_path = os.path.join(root, dir_name)
                self.hour = dir_name[-2:]
                self.parent_folder_name = dir_name
                self.year_month = dir_name[:6]
                if self.parent_folder_name == self.target_date:
                    logging.info(f"Processing folder: {dir_name}")
                    self.start_stamp = int(
                        datetime.strptime(dir_name, "%Y%m%d%H").timestamp()
                    )
                    self.process_folder(folder_path)
                    # self.clear()

    # def clear(self):
    #     os.execv(sys.executable, [sys.executable] + sys.argv)

    # 记录已经转换的起报时间
    def save_records(self, records):
        """将记录（列表）写入文件"""
        with open(self.start_times_load_file_path, "w", encoding="utf8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    # 读取本地记录数据
    def load_records(self):
        """从文件中读取记录，返回列表"""
        with open(self.start_times_load_file_path, "r", encoding="utf8") as f:
            return json.load(f)

    def write_all_zarr(self):
        """
        使用线程池并行写入多个 key 的数据。
        """
        all_data_dict = self.all_data.get_all()

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {
                executor.submit(self.write_zarr_for_key, key, data_dict): key
                for key, data_dict in all_data_dict.items()
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error writing key {futures[future]}: {e}")
        logging.info("所有数据写入成功!")
        self.start_times_list.append(self.parent_folder_name)
        self.save_records(self.start_times_list)

    # 在 write_zarr_for_key 中修改写入部分，例如：
    def write_zarr_for_key(self, name, data_dict):
        """
        独立线程内写入 Zarr 数据，确保数据具有统一内层尺寸（例如 (80,202,247)），
        保证写入的稳定性和通用性。
        """
        try:
            # 解析数据
            forecast_times = data_dict["forecast"]  # 预测时间列表

            if len(forecast_times) < self.count:
                # 计算需要补充的个数
                num_to_add = self.count - len(forecast_times)
                # 扩展列表
                forecast_times.extend([np.nan] * num_to_add)

            data = data_dict["data"]  # 每个预测时间的数据列表
            data_array = np.stack(data, axis=0)  # 例如 shape 为 (n, H, W)

            # 统一内层数据尺寸到预期 (80,202,247)
            data_array = self.standardize_array(data_array)

            # 获取统一后的尺寸
            shape0, shape1, shape2 = data_array.shape  # 现在 shape0 一定为80
            hpa_surf = data_dict["hpa_surf"]
            start_time = self.start_stamp

            # 生成 Zarr 存储路径（目录保持不变）
            zarr_path = os.path.join(
                self.target_dir,
                "EC",
                "atmos",
                self.hour,
                str(hpa_surf),
                name,
                self.year_month,
            )
            os.makedirs(zarr_path, exist_ok=True)

            # 线程内部创建 store，避免主线程对象锁问题
            store = zarr.DirectoryStore(zarr_path)
            chunk_size = (1, self.count, self.shape1, self.shape2)  # 固定内层尺寸

            if os.path.exists(os.path.join(zarr_path, ".zarray")):
                # 打开现有数组
                zarr_array = zarr.open(store, mode="a")
                # 获取现有元数据
                start_times = zarr_array.attrs.get("start_times", [])
                forecast_times_all = zarr_array.attrs.get("forecast_times", [])
            else:
                # 定义新 Zarr 数组，初始第二维度大小为0（后续动态扩展）
                zarr_array = zarr.open(
                    store,
                    mode="a",
                    shape=(0, self.count, self.shape1, self.shape2),
                    chunks=chunk_size,
                    dtype="f4",
                )
                start_times = []
                forecast_times_all = []
            if start_time not in start_times:
                # 追加当前起报时间及预测时间信息
                start_times.append(start_time)
                forecast_times_all.append(forecast_times)

                # 统一扩充数据，在写入时将 data_array 的形状扩展为 (1, shape0, shape1, shape2)
                data_to_write = data_array.reshape(
                    (1, self.count, self.shape1, self.shape2)
                )
                zarr_array.append(data_to_write, axis=0)

                attrs_to_update = {}
                attrs_to_update["start_times"] = start_times  # 确保时间数据为列表
                attrs_to_update["forecast_times"] = forecast_times_all
                # 如果属性不存在则添加经纬度属性
                if "longitudes" not in zarr_array.attrs:
                    attrs_to_update["longitudes"] = list(self.longitudes)
                if "latitudes" not in zarr_array.attrs:
                    attrs_to_update["latitudes"] = list(self.latitudes)
                # 更新元数据
                zarr_array.attrs.update(attrs_to_update)
            else:
                # 追加当前起报时间及预测时间信息
                idx = start_times.index(start_time)

                forecast_times_all[idx] = forecast_times

                # 统一扩充数据，在写入时将 data_array 的形状扩展为 (1, shape0, shape1, shape2)
                # data_to_write = data_array.reshape((1, self.count, self.shape1, self.shape2))
                zarr_array[idx] = data_array

                attrs_to_update = {}
                attrs_to_update["start_times"] = start_times  # 确保时间数据为列表
                attrs_to_update["forecast_times"] = forecast_times_all
                # 如果属性不存在则添加经纬度属性
                if "longitudes" not in zarr_array.attrs:
                    attrs_to_update["longitudes"] = list(self.longitudes)
                if "latitudes" not in zarr_array.attrs:
                    attrs_to_update["latitudes"] = list(self.latitudes)
                # 更新元数据
                zarr_array.attrs.update(attrs_to_update)

        except Exception as e:
            logging.error(f"{name}'s start time already exists. {e}")

    def standardize_array(self, data_array):
        """
        如果 data_array 的形状与 expected_shape 不一致，
        则对第一维进行 pad 或 trim，以获得固定形状 expected_shape。
        使用 np.nan 填充缺失部分。
        """
        expected_shape = (self.count, self.shape1, self.shape2)
        current_shape = data_array.shape  # 例如 (17, 202, 247) 或 (80,202,247)
        expected_first_dim = expected_shape[0]

        if current_shape[0] < expected_first_dim:
            # 填充缺失的部分，用 np.nan 填充
            pad_width = [(0, expected_first_dim - current_shape[0])] + [(0, 0)] * (
                data_array.ndim - 1
            )
            data_array = np.pad(
                data_array, pad_width, mode="constant", constant_values=np.nan
            )
        elif current_shape[0] > expected_first_dim:
            # 超出部分直接截断
            data_array = data_array[:expected_first_dim]

        return data_array


if __name__ == "__main__":
    root_dir = "/home/admin123/EC_Atmos_01/Data"
    target_dir = "/home/admin123/Data3/Zarr/China"

    date = "2025033112"

    processor = WeatherDataProcessor(root_dir, target_dir, date, max_threads=17)
    processor.process_files()
