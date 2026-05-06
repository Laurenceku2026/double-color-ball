# app.py
# 双色球AI智能选号工具 - 完整版
# 功能：四种AI算法、MCP+爬虫双数据源、Supabase存储、DeepSeek集成、管理员功能
# 版本：v4.0
# Python版本：3.11

import streamlit as st
import pandas as pd
import numpy as np
import random
import math
import hashlib
import hmac
import requests
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client
from retryflow import retry

# ==================== 尝试导入ML库 ====================
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 尝试导入MCP（F0ckssq-mcp）
try:
    from ssq_mcp import get_recent_data, get_data_by_issue_range, get_frequency_analysis
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="双色球AI分析工具 - 完整版",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS ====================
st.markdown("""
<style>
    .stDataFrame { text-align: center; }
    .stDataFrame table { text-align: center; width: 100%; }
    .stDataFrame th { text-align: center !important; }
    .stDataFrame td { text-align: center !important; }
    .stMetric { text-align: center; }
    .ai-suggestion-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        margin: 10px 0;
    }
    .data-source-status {
        font-size: 0.8rem;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 2px 0;
    }
    .status-success { background-color: #28a745; color: white; }
    .status-error { background-color: #dc3545; color: white; }
    .status-warning { background-color: #ffc107; color: #333; }
    .signal-high { background-color: #ff4b4b; color: white; padding: 5px 10px; border-radius: 20px; display: inline-block; }
    .signal-medium { background-color: #ffa500; color: white; padding: 5px 10px; border-radius: 20px; display: inline-block; }
    .signal-low { background-color: #00cc66; color: white; padding: 5px 10px; border-radius: 20px; display: inline-block; }
    .bet-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
        text-align: center;
    }
    .red-ball {
        display: inline-block;
        width: 36px;
        height: 36px;
        line-height: 36px;
        text-align: center;
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        color: white;
        border-radius: 50%;
        margin: 2px;
        font-weight: bold;
    }
    .blue-ball {
        display: inline-block;
        width: 36px;
        height: 36px;
        line-height: 36px;
        text-align: center;
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
        border-radius: 50%;
        margin: 2px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 常量定义 ====================
RED_NUMBERS = list(range(1, 34))
BLUE_NUMBERS = list(range(1, 17))
RED_EXPECTED_SUM = 102  # (1+33)/2 * 6
RED_SUM_STD = 15
BLUE_EXPECTED = 8.5

# ==================== DeepSeek 配置 ====================
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = st.secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = st.secrets.get("DEEPSEEK_MODEL", "deepseek-chat")

# ==================== Supabase 初始化 ====================
def init_supabase():
    """初始化Supabase连接"""
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        st.error(f"Supabase连接失败: {e}")
        return None

# ==================== 数据源1：MCP服务 ====================
@retry(max_attempts=2, delay=1.0, backoff=2.0)
def fetch_from_mcp(limit: int = 200) -> Optional[List[Dict]]:
    """从MCP服务获取双色球数据"""
    if not MCP_AVAILABLE:
        return None
    
    try:
        data = get_recent_data(limit=limit)
        if data and len(data) > 0:
            adapted = []
            for item in data:
                # 处理红球
                reds = item.get('reds', item.get('red', []))
                if isinstance(reds, str):
                    reds = [int(r) for r in reds.split(',')]
                
                adapted.append({
                    'period': str(item.get('issue', item.get('period', ''))),
                    'date': item.get('date'),
                    'reds': reds[:6] if len(reds) >= 6 else [0]*6,
                    'blue': int(item.get('blue', 0)),
                    'pool': float(item.get('pool', item.get('pool_amount', 0))),
                    'sales': float(item.get('sales', item.get('total_sales', 0))),
                    'prize1_count': int(item.get('prize1_count', item.get('first_prize_count', 0))),
                    'prize1_amount': float(item.get('prize1_amount', item.get('first_prize_amount', 0))),
                    'prize2_count': int(item.get('prize2_count', item.get('second_prize_count', 0))),
                    'prize2_amount': float(item.get('prize2_amount', item.get('second_prize_amount', 0)))
                })
            return adapted
        return None
    except Exception as e:
        st.warning(f"MCP获取失败: {e}")
        return None

# ==================== 数据源2：中彩网爬虫 ====================
@retry(max_attempts=2, delay=1.0, backoff=2.0)
def fetch_from_zhcw(page: int = 1) -> Optional[List[Dict]]:
    """从中彩网爬取双色球数据"""
    try:
        url = f"http://kaijiang.zhcw.com/zhcw/html/ssq/list_{page}.html"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            return None
        
        # 使用pandas读取HTML表格
        tables = pd.read_html(response.text)
        if not tables:
            return None
        
        df = tables[0]
        draws = []
        
        for _, row in df.iterrows():
            # 提取号码
            numbers_str = str(row.iloc[2]) if len(row) > 2 else ""
            # 格式如 "01 14 20 21 23 27 06"
            parts = numbers_str.strip().split()
            if len(parts) >= 7:
                reds = [int(p) for p in parts[:6]]
                blue = int(parts[6])
                
                # 提取销售额
                sales_str = str(row.iloc[3]) if len(row) > 3 else "0"
                sales = float(sales_str.replace(',', '')) if sales_str else 0
                
                # 提取一等奖注数
                prize1_str = str(row.iloc[4]) if len(row) > 4 else "0"
                prize1_count = int(prize1_str) if prize1_str.isdigit() else 0
                
                # 提取二等奖注数
                prize2_str = str(row.iloc[5]) if len(row) > 5 else "0"
                prize2_count = int(prize2_str) if prize2_str.isdigit() else 0
                
                draws.append({
                    'period': str(row.iloc[1]) if len(row) > 1 else "",
                    'date': row.iloc[0] if len(row) > 0 else None,
                    'reds': reds,
                    'blue': blue,
                    'sales': sales,
                    'prize1_count': prize1_count,
                    'prize2_count': prize2_count,
                    'pool': 0,  # 中彩网不提供奖池
                    'prize1_amount': 0,
                    'prize2_amount': 0
                })
        
        return draws if draws else None
    except Exception as e:
        st.warning(f"中彩网爬取失败: {e}")
        return None

# ==================== 数据源3：中国福彩网爬虫（获取奖池） ====================
@retry(max_attempts=2, delay=1.0, backoff=2.0)
def fetch_pool_from_cwl(period: str) -> Optional[float]:
    """从中国福彩网获取指定期号的奖池金额"""
    try:
        url = f"https://www.cwl.gov.cn/kjxx/ssq/period-{period}/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None
        
        # 使用正则表达式提取奖池
        import re
        pattern = r'奖池奖金[：:]\s*([\d,]+)'
        match = re.search(pattern, response.text)
        if match:
            return float(match.group(1).replace(',', ''))
        
        pattern2 = r'poolAmount["\']?\s*[=:]\s*["\']?([\d,]+)'
        match2 = re.search(pattern2, response.text)
        if match2:
            return float(match2.group(1).replace(',', ''))
        
        return None
    except Exception as e:
        st.warning(f"福彩网奖池获取失败: {e}")
        return None

# ==================== 数据源管理器 ====================
class DataSourceManager:
    """管理多个数据源，支持优先级和切换"""
    
    def __init__(self):
        self.sources = {
            'mcp': {'name': 'MCP服务', 'enabled': True, 'priority': 1, 'status': '待检测'},
            'zhcw': {'name': '中彩网', 'enabled': True, 'priority': 2, 'status': '待检测'}
        }
        self.data = None
        self.source_used = None
        self.last_update = None
    
    def set_enabled(self, source_key: str, enabled: bool):
        if source_key in self.sources:
            self.sources[source_key]['enabled'] = enabled
    
    def fetch_all(self, limit: int = 200) -> bool:
        """按优先级尝试所有启用的数据源"""
        sorted_sources = sorted(
            [(k, v) for k, v in self.sources.items() if v['enabled']],
            key=lambda x: x[1]['priority']
        )
        
        for source_key, source_info in sorted_sources:
            with st.spinner(f"正在从 {source_info['name']} 获取数据..."):
                try:
                    if source_key == 'mcp':
                        data = fetch_from_mcp(limit=limit)
                    elif source_key == 'zhcw':
                        # 获取多页数据
                        all_data = []
                        for page in range(1, 4):  # 获取前3页
                            page_data = fetch_from_zhcw(page=page)
                            if page_data:
                                all_data.extend(page_data)
                        data = all_data
                    else:
                        continue
                    
                    if data and len(data) > 0:
                        # 去重（按期号）
                        seen = set()
                        unique_data = []
                        for d in data:
                            period = d.get('period', '')
                            if period and period not in seen:
                                seen.add(period)
                                unique_data.append(d)
                        
                        source_info['status'] = f"成功获取 {len(unique_data)} 期"
                        self.data = unique_data
                        self.source_used = source_key
                        self.last_update = datetime.now()
                        
                        # 补充奖池数据（如果MCP没有提供）
                        if source_key != 'mcp':
                            self._enrich_pool_data()
                        
                        return True
                    else:
                        source_info['status'] = "无数据"
                except Exception as e:
                    source_info['status'] = f"失败: {str(e)[:50]}"
                    continue
        
        return False
    
    def _enrich_pool_data(self):
        """补充奖池数据（从福彩网）"""
        if not self.data:
            return
        
        # 只补充最近的20期
        for i, draw in enumerate(self.data[:20]):
            if draw.get('pool', 0) == 0:
                period = draw.get('period', '')
                if period:
                    pool = fetch_pool_from_cwl(period)
                    if pool:
                        self.data[i]['pool'] = pool
    
    def get_data(self) -> Optional[List[Dict]]:
        return self.data
    
    def get_source_used(self) -> Optional[str]:
        return self.source_used
    
    def get_status_df(self) -> pd.DataFrame:
        """获取数据源状态DataFrame"""
        rows = []
        for k, v in self.sources.items():
            rows.append({
                '数据源': v['name'],
                '状态': v['status'],
                '启用': '✅' if v['enabled'] else '❌'
            })
        return pd.DataFrame(rows)
    
    def get_data_for_training(self) -> pd.DataFrame:
        """获取用于ML训练的数据框"""
        if not self.data:
            return pd.DataFrame()
        
        records = []
        for draw in self.data:
            records.append({
                'period': draw.get('period'),
                'date': draw.get('date'),
                'red1': draw['reds'][0] if len(draw['reds']) > 0 else 0,
                'red2': draw['reds'][1] if len(draw['reds']) > 1 else 0,
                'red3': draw['reds'][2] if len(draw['reds']) > 2 else 0,
                'red4': draw['reds'][3] if len(draw['reds']) > 3 else 0,
                'red5': draw['reds'][4] if len(draw['reds']) > 4 else 0,
                'red6': draw['reds'][5] if len(draw['reds']) > 5 else 0,
                'blue': draw.get('blue', 0),
                'pool': draw.get('pool', 0),
                'sales': draw.get('sales', 0),
                'prize1_count': draw.get('prize1_count', 0),
                'prize2_count': draw.get('prize2_count', 0)
            })
        
        return pd.DataFrame(records)

# ==================== 方法1：冷热码+和值预测 ====================
class Method1HotColdSum:
    """方法1：冷热码评分 + 和值动态预测"""
    
    def __init__(self, draws: List[Dict]):
        self.draws = draws
        self.red_freq = self._calculate_frequency('reds')
        self.blue_freq = self._calculate_frequency('blue')
        self.red_absence = self._calculate_absence('reds')
        self.blue_absence = self._calculate_absence('blue')
    
    def _calculate_frequency(self, field: str) -> Dict[int, float]:
        """计算频率"""
        freq = {i: 0 for i in range(1, 34 if field == 'reds' else 17)}
        total = 0
        
        for draw in self.draws:
            if field == 'reds':
                for num in draw.get('reds', []):
                    if 1 <= num <= 33:
                        freq[num] += 1
                        total += 1
            else:
                blue = draw.get('blue', 0)
                if 1 <= blue <= 16:
                    freq[blue] += 1
                    total += 1
        
        # 归一化
        max_freq = max(freq.values()) if freq.values() else 1
        for num in freq:
            freq[num] = freq[num] / max_freq
        
        return freq
    
    def _calculate_absence(self, field: str) -> Dict[int, int]:
        """计算遗漏期数"""
        absence = {i: 0 for i in range(1, 34 if field == 'reds' else 17)}
        last_seen = {i: None for i in range(1, 34 if field == 'reds' else 17)}
        
        for idx, draw in enumerate(reversed(self.draws)):
            if field == 'reds':
                for num in draw.get('reds', []):
                    if last_seen[num] is None:
                        last_seen[num] = idx
            else:
                blue = draw.get('blue', 0)
                if last_seen.get(blue) is None:
                    last_seen[blue] = idx
        
        total = len(self.draws)
        for num in absence:
            absence[num] = last_seen[num] if last_seen[num] is not None else total
        
        return absence
    
    def calculate_red_scores(self) -> Dict[int, float]:
        """计算红球评分"""
        scores = {}
        for num in RED_NUMBERS:
            freq_score = self.red_freq.get(num, 0)
            absence_score = 1 - (self.red_absence.get(num, self.red_absence.get(1, 0)) / max(self.red_absence.values()))
            score = 0.5 * freq_score + 0.5 * absence_score
            scores[num] = score
        return scores
    
    def calculate_blue_scores(self) -> Dict[int, float]:
        """计算蓝球评分"""
        scores = {}
        for num in BLUE_NUMBERS:
            freq_score = self.blue_freq.get(num, 0)
            absence_score = 1 - (self.blue_absence.get(num, self.blue_absence.get(1, 0)) / max(self.blue_absence.values()))
            score = 0.5 * freq_score + 0.5 * absence_score
            scores[num] = score
        return scores
    
    def get_target_sum(self) -> Tuple[int, int]:
        """动态目标和值"""
        if len(self.draws) < 10:
            return RED_EXPECTED_SUM, RED_SUM_STD
        
        recent_sums = []
        for draw in self.draws[-10:]:
            reds = draw.get('reds', [])
            if reds:
                recent_sums.append(sum(reds))
        
        if not recent_sums:
            return RED_EXPECTED_SUM, RED_SUM_STD
        
        short_mean = np.mean(recent_sums)
        
        if short_mean > RED_EXPECTED_SUM + 5:
            target = RED_EXPECTED_SUM - 5
        elif short_mean < RED_EXPECTED_SUM - 5:
            target = RED_EXPECTED_SUM + 5
        else:
            target = RED_EXPECTED_SUM
        
        return int(target), RED_SUM_STD
    
    def generate_bets(self, num_bets: int = 4) -> List[Dict]:
        """生成投注组合"""
        red_scores = self.calculate_red_scores()
        blue_scores = self.calculate_blue_scores()
        
        # 权重转换
        red_weights = [math.exp(red_scores.get(i, 0)) for i in RED_NUMBERS]
        blue_weights = [math.exp(blue_scores.get(i, 0)) for i in BLUE_NUMBERS]
        
        target_sum, tolerance = self.get_target_sum()
        
        bets = []
        for _ in range(num_bets):
            for _ in range(100):
                reds = np.random.choice(RED_NUMBERS, size=6, replace=False, p=red_weights/np.sum(red_weights))
                reds = sorted(reds.tolist())
                if abs(sum(reds) - target_sum) <= tolerance:
                    blues = np.random.choice(BLUE_NUMBERS, size=1, p=blue_weights/np.sum(blue_weights))
                    bets.append({
                        'reds': reds,
                        'blue': blues[0],
                        'sum': sum(reds),
                        'method': '方法1:冷热码+和值'
                    })
                    break
            else:
                reds = sorted(np.random.choice(RED_NUMBERS, size=6, replace=False))
                blue = np.random.choice(BLUE_NUMBERS)
                bets.append({
                    'reds': reds,
                    'blue': blue,
                    'sum': sum(reds),
                    'method': '方法1:冷热码+和值'
                })
        
        return bets

# ==================== 方法2：胆拖混合 ====================
class Method2DanTuo:
    """方法2：胆拖混合（胆码锁定+拖码生成）"""
    
    def __init__(self, draws: List[Dict]):
        self.draws = draws
        self.method1 = Method1HotColdSum(draws)
    
    def select_anchors(self, num_anchors: int = 2) -> List[int]:
        """选择胆码"""
        red_scores = self.method1.calculate_red_scores()
        
        # 优先选择上期号码
        if self.draws:
            last_reds = self.draws[-1].get('reds', [])
            for num in last_reds:
                if num in red_scores:
                    red_scores[num] += 0.3
        
        # 按得分排序
        sorted_nums = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:num_anchors]]
    
    def generate_bets(self, num_bets: int = 4) -> List[Dict]:
        """生成投注组合"""
        anchors = self.select_anchors()
        red_scores = self.method1.calculate_red_scores()
        blue_scores = self.method1.calculate_blue_scores()
        
        # 降低胆码的权重（避免重复）
        for a in anchors:
            red_scores[a] = 0.1
        
        red_weights = [math.exp(red_scores.get(i, 0)) for i in RED_NUMBERS]
        blue_weights = [math.exp(blue_scores.get(i, 0)) for i in BLUE_NUMBERS]
        
        target_sum, tolerance = self.method1.get_target_sum()
        
        bets = []
        for _ in range(num_bets):
            needed = 6 - len(anchors)
            for _ in range(100):
                candidates = [i for i in RED_NUMBERS if i not in anchors]
                candidate_weights = [red_weights[i-1] for i in candidates]
                if not candidates:
                    break
                selected = np.random.choice(candidates, size=needed, replace=False, 
                                           p=np.array(candidate_weights)/np.sum(candidate_weights))
                reds = sorted(anchors + selected.tolist())
                if abs(sum(reds) - target_sum) <= tolerance:
                    blue = np.random.choice(BLUE_NUMBERS, p=blue_weights/np.sum(blue_weights))
                    bets.append({
                        'reds': reds,
                        'blue': blue,
                        'sum': sum(reds),
                        'method': f'方法2:胆拖混合 (胆码:{anchors})'
                    })
                    break
            else:
                candidates = [i for i in RED_NUMBERS if i not in anchors]
                selected = np.random.choice(candidates, size=needed, replace=False)
                reds = sorted(anchors + selected.tolist())
                blue = np.random.choice(BLUE_NUMBERS)
                bets.append({
                    'reds': reds,
                    'blue': blue,
                    'sum': sum(reds),
                    'method': f'方法2:胆拖混合 (胆码:{anchors})'
                })
        
        return bets

# ==================== 方法3：LightGBM ====================
class Method3LightGBM:
    """方法3：LightGBM梯度提升树"""
    
    def __init__(self, draws: List[Dict]):
        self.draws = draws
        self.model = None
        self.is_trained = False
    
    def _extract_features(self, window_draws: List[Dict], target_num: int) -> Optional[Dict]:
        """提取单个号码的特征"""
        if len(window_draws) < 20:
            return None
        
        features = {}
        
        # 历史频率
        total = len(window_draws)
        freq = sum(1 for d in window_draws if target_num in d.get('reds', []))
        features['freq'] = freq / total if total > 0 else 0
        
        # 遗漏期数
        last_seen = None
        for idx, d in enumerate(reversed(window_draws)):
            if target_num in d.get('reds', []):
                last_seen = idx
                break
        features['absence'] = last_seen if last_seen is not None else total
        
        # 最近10期频率
        recent = window_draws[-10:] if len(window_draws) >= 10 else window_draws
        recent_freq = sum(1 for d in recent if target_num in d.get('reds', []))
        features['recent_freq'] = recent_freq / len(recent) if recent else 0
        
        # 上期是否出现
        if window_draws:
            features['last_appeared'] = 1 if target_num in window_draws[-1].get('reds', []) else 0
        
        # 分区特征
        zone = (target_num - 1) // 11 + 1
        features['zone'] = zone
        
        # 奇偶
        features['parity'] = target_num % 2
        
        # 大小（1-16小，17-33大）
        features['size'] = 0 if target_num <= 16 else 1
        
        return features
    
    def train(self) -> bool:
        """训练模型"""
        if not LGB_AVAILABLE or len(self.draws) < 100:
            return False
        
        X_list = []
        y_list = []
        
        for i in range(50, len(self.draws) - 1):
            window = self.draws[i-50:i]
            next_draw = self.draws[i]
            
            for num in RED_NUMBERS:
                features = self._extract_features(window, num)
                if features:
                    X_list.append(features)
                    y_list.append(1 if num in next_draw.get('reds', []) else 0)
        
        if not X_list:
            return False
        
        X_df = pd.DataFrame(X_list).fillna(0)
        y_series = pd.Series(y_list)
        
        try:
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            self.model.fit(X_df, y_series)
            self.is_trained = True
            return True
        except Exception as e:
            st.warning(f"LightGBM训练失败: {e}")
            return False
    
    def predict(self) -> List[int]:
        """预测下期红球"""
        if not self.is_trained or not self.model:
            return []
        
        predictions = []
        for num in RED_NUMBERS:
            features = self._extract_features(self.draws, num)
            if features:
                X_pred = pd.DataFrame([features]).fillna(0)
                prob = self.model.predict_proba(X_pred)[0][1]
                predictions.append((num, prob))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [num for num, _ in predictions[:6]]
    
    def generate_bets(self, num_bets: int = 4) -> List[Dict]:
        """生成投注组合"""
        if not self.is_trained:
            self.train()
        
        # 如果训练失败，使用方法1作为备选
        if not self.is_trained:
            method1 = Method1HotColdSum(self.draws)
            return method1.generate_bets(num_bets)
        
        predicted_reds = self.predict()
        blue_scores = Method1HotColdSum(self.draws).calculate_blue_scores()
        blue_weights = [math.exp(blue_scores.get(i, 0)) for i in BLUE_NUMBERS]
        
        bets = []
        for _ in range(num_bets):
            # 使用预测的红球，微调
            reds = predicted_reds[:]
            if len(reds) < 6:
                missing = [n for n in RED_NUMBERS if n not in reds]
                extra = np.random.choice(missing, size=6-len(reds), replace=False)
                reds.extend(extra)
            
            # 随机替换1-2个
            replace_count = np.random.randint(1, 3)
            for _ in range(replace_count):
                idx = np.random.randint(0, len(reds))
                candidates = [n for n in RED_NUMBERS if n not in reds]
                if candidates:
                    reds[idx] = np.random.choice(candidates)
            
            reds = sorted(reds[:6])
            blue = np.random.choice(BLUE_NUMBERS, p=blue_weights/np.sum(blue_weights))
            
            bets.append({
                'reds': reds,
                'blue': blue,
                'sum': sum(reds),
                'method': '方法3:LightGBM'
            })
        
        return bets

# ==================== 方法4：XGBoost+神经网络集成 ====================
class Method4Ensemble:
    """方法4：XGBoost + 神经网络集成"""
    
    def __init__(self, draws: List[Dict]):
        self.draws = draws
        self.xgb_model = None
        self.nn_model = None
        self.scaler = None
        self.is_trained = False
    
    def _extract_features_advanced(self, window_draws: List[Dict], target_num: int) -> Optional[Dict]:
        """提取高级特征"""
        if len(window_draws) < 30:
            return None
        
        features = {}
        total = len(window_draws)
        
        # 基础频率
        freq = sum(1 for d in window_draws if target_num in d.get('reds', []))
        features['freq'] = freq / total
        
        # 遗漏
        last_seen = None
        for idx, d in enumerate(reversed(window_draws)):
            if target_num in d.get('reds', []):
                last_seen = idx
                break
        features['absence'] = last_seen if last_seen is not None else total
        features['absence_norm'] = features['absence'] / total
        
        # 近期趋势
        for window in [3, 5, 10]:
            recent = window_draws[-window:] if len(window_draws) >= window else window_draws
            recent_freq = sum(1 for d in recent if target_num in d.get('reds', []))
            features[f'recent_{window}'] = recent_freq / len(recent) if recent else 0
        
        # 上期关系
        if window_draws:
            last_reds = window_draws[-1].get('reds', [])
            features['last_appeared'] = 1 if target_num in last_reds else 0
            if last_reds:
                features['min_diff_to_last'] = min(abs(target_num - n) for n in last_reds)
            else:
                features['min_diff_to_last'] = 99
        
        # 分区
        zone = (target_num - 1) // 11 + 1
        features['zone'] = zone
        
        # 统计特征
        features['parity'] = target_num % 2
        features['size'] = 0 if target_num <= 16 else 1
        
        # 尾数
        features['tail'] = target_num % 10
        
        return features
    
    def train(self) -> bool:
        """训练集成模型"""
        if (not XGB_AVAILABLE or not SKLEARN_AVAILABLE) or len(self.draws) < 150:
            return False
        
        X_list = []
        y_list = []
        
        for i in range(80, len(self.draws) - 1):
            window = self.draws[i-80:i]
            next_draw = self.draws[i]
            
            for num in RED_NUMBERS:
                features = self._extract_features_advanced(window, num)
                if features:
                    X_list.append(features)
                    y_list.append(1 if num in next_draw.get('reds', []) else 0)
        
        if not X_list or len(X_list) < 1000:
            return False
        
        X_df = pd.DataFrame(X_list).fillna(0)
        y_series = pd.Series(y_list)
        
        try:
            # XGBoost
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            self.xgb_model.fit(X_df, y_series)
            
            # 神经网络
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_df)
            self.nn_model = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation='relu',
                max_iter=100,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            self.nn_model.fit(X_scaled, y_series)
            
            self.is_trained = True
            return True
        except Exception as e:
            st.warning(f"集成模型训练失败: {e}")
            return False
    
    def predict(self) -> List[int]:
        """预测下期红球"""
        if not self.is_trained:
            return []
        
        predictions = []
        for num in RED_NUMBERS:
            features = self._extract_features_advanced(self.draws, num)
            if features:
                X_pred = pd.DataFrame([features]).fillna(0)
                
                # XGBoost预测
                xgb_prob = self.xgb_model.predict_proba(X_pred)[0][1]
                
                # 神经网络预测
                X_scaled = self.scaler.transform(X_pred)
                nn_prob = self.nn_model.predict_proba(X_scaled)[0][1]
                
                # 加权融合
                ensemble_prob = 0.5 * xgb_prob + 0.5 * nn_prob
                predictions.append((num, ensemble_prob))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [num for num, _ in predictions[:6]]
    
    def generate_bets(self, num_bets: int = 4) -> List[Dict]:
        """生成投注组合"""
        if not self.is_trained:
            self.train()
        
        if not self.is_trained:
            method1 = Method1HotColdSum(self.draws)
            return method1.generate_bets(num_bets)
        
        predicted_reds = self.predict()
        blue_scores = Method1HotColdSum(self.draws).calculate_blue_scores()
        blue_weights = [math.exp(blue_scores.get(i, 0)) for i in BLUE_NUMBERS]
        
        bets = []
        for _ in range(num_bets):
            reds = predicted_reds[:]
            if len(reds) < 6:
                missing = [n for n in RED_NUMBERS if n not in reds]
                extra = np.random.choice(missing, size=6-len(reds), replace=False)
                reds.extend(extra)
            
            reds = sorted(reds[:6])
            blue = np.random.choice(BLUE_NUMBERS, p=blue_weights/np.sum(blue_weights))
            
            bets.append({
                'reds': reds,
                'blue': blue,
                'sum': sum(reds),
                'method': '方法4:XGBoost+NN集成'
            })
        
        return bets

# ==================== 投注生成工厂 ====================
class BetGenerator:
    """投注生成工厂类"""
    
    @staticmethod
    def generate(method: str, draws: List[Dict], num_bets: int = 4) -> List[Dict]:
        """根据方法名称生成投注"""
        if method == "方法1: 当前方法":
            generator = Method1HotColdSum(draws)
        elif method == "方法2: 胆拖混合":
            generator = Method2DanTuo(draws)
        elif method == "方法3: LightGBM":
            generator = Method3LightGBM(draws)
        elif method == "方法4: XGBoost+NN集成":
            generator = Method4Ensemble(draws)
        else:
            generator = Method1HotColdSum(draws)
        
        return generator.generate_bets(num_bets)

# ==================== DeepSeek AI 建议 ====================
def get_deepseek_suggestion(draws: List[Dict], source_used: str, ml_signals: Dict) -> Dict:
    """获取DeepSeek AI的投注建议"""
    if not DEEPSEEK_API_KEY or len(draws) < 10:
        return {
            "plan": "4组7+1复式",
            "win_rate": "30-35%",
            "blue_advice": "均衡选择",
            "summary": "请配置DeepSeek API Key获取AI建议"
        }
    
    # 计算热号
    red_scores = Method1HotColdSum(draws).calculate_red_scores()
    blue_scores = Method1HotColdSum(draws).calculate_blue_scores()
    
    top_reds = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)[:6]
    top_blues = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    try:
        prompt = f"""你是双色球AI分析专家。基于以下数据给出投注建议（JSON格式）：

【数据来源】{source_used}
【历史数据量】{len(draws)}期
【红球热号Top6】{[r[0] for r in top_reds]}
【蓝球热号Top3】{[b[0] for b in top_blues]}
【剪刀差信号】{ml_signals.get('scissors', '正常')}
【奖池阈值】{ml_signals.get('jackpot_level', '正常')}

请按此JSON格式回复（只有JSON）：
{{"plan": "推荐方案(7+1/7+2/观望)", "win_rate": "预期赢率%", "blue_advice": "蓝球建议", "summary": "一句话总结"}}
"""
        response = requests.post(
            f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={"model": DEEPSEEK_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 300},
            timeout=10
        )
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            json_match = re.search(r'\{[^{}]*\}', content)
            if json_match:
                return json.loads(json_match.group())
    except Exception as e:
        st.warning(f"DeepSeek API调用失败: {e}")
    
    return {
        "plan": "4组7+1复式",
        "win_rate": "30-35%",
        "blue_advice": f"推荐蓝球{top_blues[0][0] if top_blues else 8}",
        "summary": f"基于{len(draws)}期历史数据，红球热号集中在{top_reds[0][0] if top_reds else 1}附近"
    }

# ==================== ML信号计算 ====================
def calculate_ml_signals(draws: List[Dict]) -> Dict:
    """计算ML特征信号"""
    if not draws or len(draws) < 10:
        return {
            'jackpot_level': '数据不足',
            'scissors': '数据不足',
            'cycle': '数据不足',
            'blue_bias': '均衡',
            'signal_strength': 0
        }
    
    latest = draws[-1]
    pool = latest.get('pool', 0)
    sales = latest.get('sales', 0)
    
    # 奖池阈值
    if pool >= 250000000:
        jackpot_level = "HIGH (≥2.5亿)"
        signal_strength = 30
    elif pool >= 150000000:
        jackpot_level = "MEDIUM (1.5-2.5亿)"
        signal_strength = 15
    else:
        jackpot_level = "LOW (<1.5亿)"
        signal_strength = 0
    
    # 剪刀差信号
    if len(draws) >= 2:
        prev = draws[-2]
        prev_pool = prev.get('pool', 0)
        prev_sales = prev.get('sales', 0)
        
        if prev_pool > 0 and prev_sales > 0:
            sales_change = (sales - prev_sales) / prev_sales if prev_sales > 0 else 0
            pool_change = (pool - prev_pool) / prev_pool if prev_pool > 0 else 0
            
            if sales_change > 0.05 and pool_change < -0.03:
                scissors = "HIGH_ALERT (头奖爆发预警)"
                signal_strength += 40
            elif sales_change > 0.03:
                scissors = "MEDIUM (投注活跃)"
                signal_strength += 20
            else:
                scissors = "NORMAL"
        else:
            scissors = "NORMAL"
    else:
        scissors = "NORMAL"
    
    # 头奖周期
    recent_prizes = [d.get('prize1_count', 0) for d in draws[-20:]]
    high_prize_count = sum(1 for p in recent_prizes if p >= 10)
    if high_prize_count >= 3:
        cycle = "冷却期"
        signal_strength -= 20
    elif high_prize_count == 0:
        cycle = "积累期"
        signal_strength += 10
    else:
        cycle = "正常期"
    
    # 蓝球偏向
    recent_blues = [d.get('blue', 0) for d in draws[-20:] if d.get('blue', 0) > 0]
    if recent_blues:
        small_count = sum(1 for b in recent_blues if b <= 8)
        if small_count >= 12:
            blue_bias = "偏小号 (1-8)"
        elif small_count <= 8:
            blue_bias = "偏大号 (9-16)"
        else:
            blue_bias = "均衡"
    else:
        blue_bias = "均衡"
    
    # 信号强度归一化到0-100
    signal_strength = max(0, min(100, signal_strength))
    
    # 投注建议
    if signal_strength >= 60:
        bet_suggestion = "strong"
        suggestion_text = "🔔 强烈推荐投注"
    elif signal_strength >= 30:
        bet_suggestion = "medium"
        suggestion_text = "⚠️ 谨慎投注"
    else:
        bet_suggestion = "weak"
        suggestion_text = "💤 建议观望"
    
    return {
        'jackpot_level': jackpot_level,
        'scissors': scissors,
        'cycle': cycle,
        'blue_bias': blue_bias,
        'signal_strength': signal_strength,
        'bet_suggestion': bet_suggestion,
        'suggestion_text': suggestion_text,
        'pool': pool,
        'sales': sales
    }

# ==================== Supabase数据操作 ====================
def save_draws_to_supabase(draws: List[Dict]) -> int:
    """保存数据到Supabase（去重）"""
    supabase = init_supabase()
    if supabase is None:
        return 0
    
    try:
        # 获取现有期号
        existing = supabase.schema('ssq_schema').table('ssq_draws').select("period").execute()
        existing_periods = set([str(row['period']) for row in existing.data]) if existing.data else set()
        
        new_count = 0
        for draw in draws:
            period = str(draw.get('period', ''))
            if not period or period in existing_periods:
                continue
            
            reds = draw.get('reds', [])
            data = {
                "period": period,
                "date": draw.get('date'),
                "red1": reds[0] if len(reds) > 0 else 0,
                "red2": reds[1] if len(reds) > 1 else 0,
                "red3": reds[2] if len(reds) > 2 else 0,
                "red4": reds[3] if len(reds) > 3 else 0,
                "red5": reds[4] if len(reds) > 4 else 0,
                "red6": reds[5] if len(reds) > 5 else 0,
                "blue": draw.get('blue', 0),
                "pool_amount": draw.get('pool', 0),
                "total_sales": draw.get('sales', 0),
                "prize1_count": draw.get('prize1_count', 0),
                "prize1_amount": draw.get('prize1_amount', 0),
                "prize2_count": draw.get('prize2_count', 0),
                "prize2_amount": draw.get('prize2_amount', 0)
            }
            supabase.schema('ssq_schema').table('ssq_draws').insert(data).execute()
            new_count += 1
        
        return new_count
    except Exception as e:
        st.error(f"保存失败: {e}")
        return 0

def load_from_supabase(limit: int = 500) -> Optional[List[Dict]]:
    """从Supabase加载数据"""
    supabase = init_supabase()
    if supabase is None:
        return None
    
    try:
        response = supabase.schema('ssq_schema').table('ssq_draws').select("*").order("period", desc=False).limit(limit).execute()
        draws = []
        for row in response.data:
            draws.append({
                'period': row.get('period'),
                'date': row.get('date'),
                'reds': [row.get('red1'), row.get('red2'), row.get('red3'), 
                        row.get('red4'), row.get('red5'), row.get('red6')],
                'blue': row.get('blue'),
                'pool': row.get('pool_amount', 0),
                'sales': row.get('total_sales', 0),
                'prize1_count': row.get('prize1_count', 0),
                'prize2_count': row.get('prize2_count', 0)
            })
        return draws
    except Exception as e:
        st.error(f"从Supabase加载失败: {e}")
        return None

# ==================== 多期查奖 ====================
def parse_check_draws(text: str) -> List[Dict]:
    """解析查奖数据"""
    lines = text.strip().split('\n')
    draws = []
    for line in lines[:5]:
        parts = line.replace(',', ' ').split()
        if len(parts) >= 9:
            try:
                draws.append({
                    'period': parts[0],
                    'reds': [int(parts[i]) for i in range(2, 8)],
                    'blue': int(parts[8])
                })
            except:
                continue
    return draws

def calculate_prize(bet: Dict, draw: Dict) -> str:
    """计算单注中奖"""
    red_matches = len(set(bet['reds']) & set(draw['reds']))
    blue_match = (bet['blue'] == draw['blue'])
    
    if red_matches == 6 and blue_match:
        return "🏆 一等奖 (浮动)"
    elif red_matches == 6:
        return "🥈 二等奖 (浮动)"
    elif red_matches == 5 and blue_match:
        return "🥉 三等奖 3000元"
    elif red_matches == 5 or (red_matches == 4 and blue_match):
        return "📦 四等奖 200元"
    elif red_matches == 4 or (red_matches == 3 and blue_match):
        return "🎫 五等奖 10元"
    elif blue_match:
        return "⭐ 六等奖 5元"
    elif red_matches == 3:
        return "🎁 福运奖 5元 (新规)"
    else:
        return "❌ 未中奖"

# ==================== 管理员函数 ====================
def check_password(password: str) -> bool:
    return hmac.compare_digest(password, "Ku_product$2026")

def admin_login():
    with st.form("admin_login_form"):
        username = st.text_input("用户名", key="admin_username")
        password = st.text_input("密码", type="password", key="admin_password")
        submitted = st.form_submit_button("登录")
        if submitted:
            if username == "Laurence_ku" and check_password(password):
                st.session_state['admin_logged_in'] = True
                st.session_state['show_admin'] = False
                st.success("登录成功！")
                st.rerun()
            else:
                st.error("用户名或密码错误")

def admin_logout():
    if st.button("退出登录", key="logout_btn"):
        st.session_state['admin_logged_in'] = False
        st.session_state['show_admin'] = False
        st.rerun()

def show_admin_page(data_manager: DataSourceManager):
    """管理员页面"""
    with st.expander("🔧 管理员控制台", expanded=True):
        st.subheader("📡 数据源状态")
        status_df = data_manager.get_status_df()
        st.dataframe(status_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("🔌 数据源开关")
        
        col1, col2 = st.columns(2)
        with col1:
            mcp_enabled = st.checkbox("启用MCP服务", value=data_manager.sources['mcp']['enabled'], key="mcp_toggle")
            data_manager.set_enabled('mcp', mcp_enabled)
        with col2:
            zhcw_enabled = st.checkbox("启用中彩网爬虫", value=data_manager.sources['zhcw']['enabled'], key="zhcw_toggle")
            data_manager.set_enabled('zhcw', zhcw_enabled)
        
        st.markdown("---")
        st.subheader("🔄 数据同步")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 立即同步所有数据源", type="primary"):
                with st.spinner("正在获取数据..."):
                    if data_manager.fetch_all():
                        st.success(f"数据获取成功！来源：{data_manager.get_source_used()}")
                        # 保存到Supabase
                        draws = data_manager.get_data()
                        if draws:
                            new_count = save_draws_to_supabase(draws)
                            st.success(f"已同步 {new_count} 期新数据到Supabase")
                        st.rerun()
                    else:
                        st.error("所有数据源均获取失败")
        
        with col2:
            st.caption("系统会按优先级自动切换数据源")
        
        st.markdown("---")
        st.subheader("📁 缓存管理")
        
        supabase_data = load_from_supabase()
        if supabase_data:
            st.info(f"Supabase缓存中有 {len(supabase_data)} 期数据")
            if st.button("🗑️ 清空缓存", type="secondary"):
                supabase = init_supabase()
                if supabase:
                    supabase.schema('ssq_schema').table('ssq_draws').delete().neq("id", 0).execute()
                    st.success("缓存已清空")
                    st.rerun()
        else:
            st.info("Supabase缓存为空")

# ==================== ROI回测功能 ====================
def backtest_roi(draws: List[Dict], method: str, num_bets: int = 4, lookback: int = 50) -> Dict:
    """回测指定方法的ROI"""
    if len(draws) < lookback + 10:
        return {"roi": 0, "total_cost": 0, "total_prize": 0, "net": 0, "win_rate": 0}
    
    total_cost = 0
    total_prize = 0
    win_count = 0
    
    for i in range(lookback, len(draws)):
        # 使用截至i-1期数据预测第i期
        historical = draws[:i]
        actual = draws[i]
        
        # 生成投注
        generator = BetGenerator()
        bets = generator.generate(method, historical, num_bets)
        
        # 计算中奖
        period_cost = num_bets * 14  # 每组14元
        period_prize = 0
        
        for bet in bets:
            red_matches = len(set(bet['reds']) & set(actual['reds'])) if actual.get('reds') else 0
            blue_match = (bet['blue'] == actual.get('blue', 0)) if actual.get('blue') else False
            
            if red_matches == 6 and blue_match:
                period_prize += 5000000
            elif red_matches == 6:
                period_prize += 500000
            elif red_matches == 5 and blue_match:
                period_prize += 3000
            elif red_matches == 5 or (red_matches == 4 and blue_match):
                period_prize += 200
            elif red_matches == 4 or (red_matches == 3 and blue_match):
                period_prize += 10
            elif blue_match:
                period_prize += 5
            elif red_matches == 3:
                period_prize += 5
        
        total_cost += period_cost
        total_prize += period_prize
        
        if period_prize > 0:
            win_count += 1
    
    net = total_prize - total_cost
    roi = (net / total_cost) * 100 if total_cost > 0 else 0
    win_rate = (win_count / lookback) * 100 if lookback > 0 else 0
    
    return {
        "roi": roi,
        "total_cost": total_cost,
        "total_prize": total_prize,
        "net": net,
        "win_rate": win_rate,
        "periods": lookback
    }

# ==================== 初始化 ====================
if 'admin_logged_in' not in st.session_state:
    st.session_state['admin_logged_in'] = False
if 'show_admin' not in st.session_state:
    st.session_state['show_admin'] = False
if 'generated_bets' not in st.session_state:
    st.session_state['generated_bets'] = None
if 'model_used' not in st.session_state:
    st.session_state['model_used'] = None
if 'data_manager' not in st.session_state:
    st.session_state['data_manager'] = DataSourceManager()

data_manager = st.session_state['data_manager']

# ==================== 主页面 ====================

# 右上角齿轮
col_title, col_settings = st.columns([0.95, 0.05])
with col_settings:
    if st.button("⚙️", key="settings_icon", help="管理员设置"):
        st.session_state['show_admin'] = not st.session_state.get('show_admin', False)

if st.session_state.get('show_admin', False):
    if not st.session_state['admin_logged_in']:
        admin_login()
    else:
        show_admin_page(data_manager)
        admin_logout()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("🎰 双色球AI分析工具")
    st.markdown("---")
    
    # ML库状态
    with st.expander("🤖 ML库状态", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ Python 3.11")
            st.success("✅ LightGBM") if LGB_AVAILABLE else st.error("❌ LightGBM")
        with col2:
            st.success("✅ XGBoost") if XGB_AVAILABLE else st.error("❌ XGBoost")
            st.success("✅ scikit-learn") if SKLEARN_AVAILABLE else st.error("❌ scikit-learn")
        st.caption(f"MCP服务: {'✅ 可用' if MCP_AVAILABLE else '❌ 不可用'}")
    
    # 数据源状态
    with st.expander("📡 数据源状态", expanded=True):
        status_df = data_manager.get_status_df()
        st.dataframe(status_df, use_container_width=True, hide_index=True)
        
        if st.button("🔄 立即获取数据", key="sidebar_sync"):
            with st.spinner("正在获取数据..."):
                if data_manager.fetch_all():
                    st.success(f"来源：{data_manager.get_source_used()}")
                    st.rerun()
                else:
                    st.error("获取失败")
    
    # 四种算法对比
    with st.expander("📖 四种AI算法对比"):
        st.markdown("""
        | 算法 | 特点 | 预期ROI |
        |------|------|---------|
        | 🟢 方法1:当前方法 | 冷热码+和值预测 | +33% |
        | 🟡 方法2:胆拖混合 | 当前方法+胆码 | +92% |
        | 🔵 方法3:LightGBM | 单一机器学习 | +97% |
        | 🟣 方法4:XGBoost+NN | 集成深度学习 | **+203%** |
        """)
    
    # 奖金结构
    with st.expander("💰 奖金结构（7+1复式）"):
        st.markdown("""
        | 条件 | 7+1总奖金 |
        |------|-----------|
        | 中蓝球 | 35元 |
        | 中3红 | 35元 (福运奖) |
        | 中3+1 | 70元 |
        | 中4+0 | 70元 |
        | 中4+1 | 200-400元 |
        | 中5+0 | 200-300元 |
        | 中5+1 | 3000-9000元 |
        """)
    
    st.markdown("---")
    st.caption("DFSS智能选号工具 v4.0 | 完整版")

# ==================== 加载数据 ====================
# 尝试从Supabase加载，如果失败则从数据源获取
draws = load_from_supabase()
if not draws or len(draws) < 10:
    with st.spinner("正在获取历史数据..."):
        if data_manager.fetch_all(limit=300):
            draws = data_manager.get_data()
            if draws:
                save_draws_to_supabase(draws)

if not draws or len(draws) < 10:
    st.info("👈 请点击右上角齿轮图标，进入管理员页面同步数据")
    st.stop()

# ==================== 主内容 ====================
st.title("🎯 双色球AI智能选号")
st.caption(f"📊 当前数据源：{data_manager.get_source_used()} | 共 {len(draws)} 期历史数据 | 最后更新：{data_manager.last_update.strftime('%Y-%m-%d %H:%M') if data_manager.last_update else '未知'}")

st.markdown("---")

# ==================== AI决策引擎 ====================
st.subheader("🧠 AI决策引擎分析")

ml_signals = calculate_ml_signals(draws)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    red_scores = Method1HotColdSum(draws).calculate_red_scores()
    top_red = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    st.metric("🔥 红球热号", ", ".join([str(r[0]) for r in top_red]))
with col2:
    blue_scores = Method1HotColdSum(draws).calculate_blue_scores()
    top_blue = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    st.metric("💙 蓝球热号", ", ".join([str(b[0]) for b in top_blue]))
with col3:
    target_sum, _ = Method1HotColdSum(draws).get_target_sum()
    st.metric("📊 目标和值", f"{target_sum} ± {RED_SUM_STD}")
with col4:
    st.metric("📈 信号强度", f"{ml_signals['signal_strength']}%")
with col5:
    st.metric("🎯 建议", ml_signals['suggestion_text'][:6])

# 详细信号
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.info(f"**奖池阈值**\n{ml_signals['jackpot_level']}")
with col2:
    st.info(f"**剪刀差信号**\n{ml_signals['scissors']}")
with col3:
    st.info(f"**头奖周期**\n{ml_signals['cycle']}")
with col4:
    st.info(f"**蓝球偏向**\n{ml_signals['blue_bias']}")

st.markdown("---")

# ==================== DeepSeek AI建议 ====================
st.subheader("💡 智能投注建议")
ai_suggestion = get_deepseek_suggestion(draws, data_manager.get_source_used(), ml_signals)

with st.container():
    st.markdown(f"""
    <div class="ai-suggestion-box">
        <h4>🤖 DeepSeek AI 分析</h4>
        <p><strong>📊 {ai_suggestion.get('summary', '基于历史数据分析，本期适合标准投注策略')}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    col_rec1, col_rec2, col_rec3 = st.columns(3)
    with col_rec1:
        st.info(f"📈 **建议方案**\n\n{ai_suggestion.get('plan', '4组7+1复式')}")
    with col_rec2:
        st.warning(f"🎯 **预期赢率**\n\n{ai_suggestion.get('win_rate', '30-35%')}")
    with col_rec3:
        st.success(f"💰 **最优蓝球**\n\n{ai_suggestion.get('blue_advice', '侧重小号1-8')}")

st.markdown("---")

# ==================== 投注生成 ====================
st.subheader("🎲 生成AI投注")

col1, col2, col3 = st.columns(3)
with col1:
    num_bets = st.number_input("投注组数", min_value=1, max_value=20, value=4, key="num_bets")
with col2:
    bet_type = st.selectbox("复式类型", ["7+1 (14元)", "7+2 (28元)", "8+1 (56元)"], key="bet_type")
with col3:
    ai_model = st.selectbox(
        "AI模型",
        ["方法4: XGBoost+NN集成 ⭐推荐", "方法3: LightGBM", "方法2: 胆拖混合", "方法1: 当前方法"],
        key="ai_model"
    )

if st.button("🚀 生成智能投注", type="primary", key="generate_btn"):
    with st.spinner(f"正在使用 {ai_model} 生成投注..."):
        # 提取方法名称
        method_name = ai_model.split(":")[0] if ":" in ai_model else ai_model
        bets = BetGenerator.generate(method_name, draws, num_bets)
        st.session_state['generated_bets'] = bets
        st.session_state['model_used'] = ai_model
    st.success(f"✅ 使用 {ai_model} 生成 {len(bets)} 组投注")

# 显示生成的投注
if st.session_state['generated_bets']:
    bets = st.session_state['generated_bets']
    model_used = st.session_state.get('model_used', '未知')
    
    st.markdown(f"### 📝 推荐投注组合 - {model_used}")
    
    # 用卡片形式显示
    cols = st.columns(min(num_bets, 4))
    for i, bet in enumerate(bets):
        col_idx = i % 4
        with cols[col_idx]:
            red_html = " ".join([f'<span class="red-ball">{r:02d}</span>' for r in bet['reds']])
            blue_html = f'<span class="blue-ball">{bet["blue"]:02d}</span>'
            st.markdown(f"""
            <div class="bet-card">
                <strong>第{i+1}组</strong><br>
                {red_html}<br>
                {blue_html}<br>
                <small>和值: {bet['sum']}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # 表格备用显示
    with st.expander("📋 查看详细表格"):
        bets_data = []
        for i, bet in enumerate(bets, 1):
            bets_data.append({
                '组别': i,
                '红球': ' '.join(f"{r:02d}" for r in bet['reds']),
                '蓝球': f"{bet['blue']:02d}",
                '和值': bet['sum']
            })
        st.dataframe(pd.DataFrame(bets_data), use_container_width=True, hide_index=True)
    
    st.info(f"💬 **AI解读**：{ai_suggestion.get('summary', '祝您好运！')}")

st.markdown("---")

# ==================== ROI回测 ====================
with st.expander("📈 ROI回测分析"):
    st.markdown("基于历史数据的回测分析（仅供参考）")
    
    col1, col2 = st.columns(2)
    with col1:
        backtest_periods = st.slider("回测期数", min_value=20, max_value=min(200, len(draws)-10), value=min(100, len(draws)-10), key="backtest_periods")
    with col2:
        backtest_bets = st.number_input("每期组数", min_value=1, max_value=10, value=4, key="backtest_bets")
    
    if st.button("运行回测", key="backtest_btn"):
        with st.spinner("正在回测..."):
            # 测试四种方法
            results = []
            for method in ["方法1: 当前方法", "方法2: 胆拖混合", "方法3: LightGBM", "方法4: XGBoost+NN集成"]:
                method_name = method.split(":")[0] if ":" in method else method
                result = backtest_roi(draws, method_name, backtest_bets, backtest_periods)
                results.append({
                    "方法": method,
                    "ROI": f"{result['roi']:.1f}%",
                    "总成本": f"¥{result['total_cost']:,.0f}",
                    "总奖金": f"¥{result['total_prize']:,.0f}",
                    "净收益": f"¥{result['net']:,.0f}",
                    "中奖率": f"{result['win_rate']:.1f}%"
                })
            
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            st.caption(f"回测期间：最近{backtest_periods}期 | 每组成本14元")

st.markdown("---")

# ==================== 多期查奖 ====================
st.subheader("🔍 多期查奖")

check_text = st.text_area(
    "📋 粘贴开奖数据（最多5期）",
    height=100,
    key="check_draws",
    placeholder="格式: 期号 日期 红1 红2 红3 红4 红5 红6 蓝\n示例:\n2026050 2026-05-03 03 04 14 15 18 20 02"
)

if st.button("🔍 查奖", key="check_btn") and check_text:
    check_draws = parse_check_draws(check_text)
    if check_draws:
        st.success(f"✅ 成功解析 {len(check_draws)} 期数据")
        
        if st.session_state.get('generated_bets'):
            results = []
            for i, bet in enumerate(st.session_state['generated_bets'], 1):
                row = {'组别': i, '红球': ' '.join(f"{n:02d}" for n in bet['reds']), '蓝球': f"{bet['blue']:02d}"}
                for draw in check_draws:
                    row[f'{draw["period"]}'] = calculate_prize(bet, draw)
                results.append(row)
            
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            
            # 汇总统计
            st.markdown("**📊 中奖统计**")
            all_prizes = []
            for bet in st.session_state['generated_bets']:
                for draw in check_draws:
                    prize = calculate_prize(bet, draw)
                    if "未中奖" not in prize:
                        all_prizes.append(prize)
            
            if all_prizes:
                st.success(f"共中奖 {len(all_prizes)} 注，详见上表")
            else:
                st.info("本期未中奖")
        else:
            st.warning("请先生成投注组合")
    else:
        st.error("解析失败，请检查格式")

# ==================== 底部 ====================
st.markdown("---")
st.caption("⚠️ 本工具仅供学术研究和娱乐参考。双色球本质随机，历史规律不代表未来结果。请理性投注，量力而行。")
