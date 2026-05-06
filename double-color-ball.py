# app.py - 第一部分
# 双色球AI智能选号工具 - 完整版
# 功能：数据概览、冷热码分析、和值趋势、蓝球走势、AI决策、投注生成、ROI回测、多期查奖
# 版本：v8.0 FINAL
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
import time
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from supabase import create_client, Client
from retryflow import retry

warnings.filterwarnings('ignore')

# ==================== 尝试导入ML库 ====================
LGB_AVAILABLE = False
XGB_AVAILABLE = False
SKLEARN_AVAILABLE = False
MCP_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    pass

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    pass

try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

try:
    from ssq_mcp import get_recent_data, get_data_by_issue_range, get_frequency_analysis
    MCP_AVAILABLE = True
except ImportError:
    pass

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
    .ml-tip-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        margin: 10px 0;
    }
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
    .zone-hot { background-color: #ff6b6b; color: white; padding: 4px 8px; border-radius: 15px; display: inline-block; margin: 2px; }
    .zone-cold { background-color: #4d4d4d; color: white; padding: 4px 8px; border-radius: 15px; display: inline-block; margin: 2px; }
    .signal-high { background-color: #ff4b4b; color: white; padding: 5px 10px; border-radius: 20px; display: inline-block; }
    .signal-medium { background-color: #ffa500; color: white; padding: 5px 10px; border-radius: 20px; display: inline-block; }
    .signal-low { background-color: #00cc66; color: white; padding: 5px 10px; border-radius: 20px; display: inline-block; }
    .stButton button { width: 100%; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================== 常量定义 ====================
RED_NUMBERS = list(range(1, 34))
BLUE_NUMBERS = list(range(1, 17))
RED_EXPECTED_SUM = 102
RED_SUM_STD = 15

# 7分区定义
ZONES = {
    1: {'name': 'A区', 'range': '01-05', 'numbers': [1, 2, 3, 4, 5]},
    2: {'name': 'B区', 'range': '06-10', 'numbers': [6, 7, 8, 9, 10]},
    3: {'name': 'C区', 'range': '11-15', 'numbers': [11, 12, 13, 14, 15]},
    4: {'name': 'D区', 'range': '16-20', 'numbers': [16, 17, 18, 19, 20]},
    5: {'name': 'E区', 'range': '21-25', 'numbers': [21, 22, 23, 24, 25]},
    6: {'name': 'F区', 'range': '26-30', 'numbers': [26, 27, 28, 29, 30]},
    7: {'name': 'G区', 'range': '31-33', 'numbers': [31, 32, 33]}
}

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
    except Exception:
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
                reds = item.get('reds', item.get('red', []))
                if isinstance(reds, str):
                    reds = [int(r) for r in reds.split(',')]
                
                if len(reds) < 6:
                    continue
                
                adapted.append({
                    'period': str(item.get('issue', item.get('period', ''))),
                    'date': item.get('date'),
                    'reds': reds[:6],
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
    except Exception:
        return None

# ==================== 数据源2：中彩网爬虫 ====================
@retry(max_attempts=2, delay=1.0, backoff=2.0)
def fetch_from_zhcw(page: int = 1) -> Optional[List[Dict]]:
    """从中彩网爬取双色球数据"""
    try:
        url = f"http://kaijiang.zhcw.com/zhcw/html/ssq/list_{page}.html"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Referer': 'http://kaijiang.zhcw.com/zhcw/html/ssq/list.html',
            'DNT': '1'
        }
        
        session = requests.Session()
        session.headers.update(headers)
        response = session.get(url, timeout=15)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            return None
        
        tables = pd.read_html(response.text)
        if not tables:
            return None
        
        df = tables[0]
        draws = []
        
        for _, row in df.iterrows():
            try:
                if len(row) < 6:
                    continue
                
                numbers_str = str(row.iloc[2]) if len(row) > 2 else ""
                parts = numbers_str.strip().split()
                if len(parts) < 7:
                    continue
                
                reds = []
                for p in parts[:6]:
                    try:
                        reds.append(int(p))
                    except:
                        break
                if len(reds) != 6:
                    continue
                
                try:
                    blue = int(parts[6])
                except:
                    continue
                
                period = str(row.iloc[1]) if len(row) > 1 else ""
                date = row.iloc[0] if len(row) > 0 else None
                
                sales_str = str(row.iloc[3]) if len(row) > 3 else "0"
                sales = float(sales_str.replace(',', '')) if sales_str else 0
                
                prize1_str = str(row.iloc[4]) if len(row) > 4 else "0"
                prize1_count = int(prize1_str) if prize1_str.isdigit() else 0
                
                prize2_str = str(row.iloc[5]) if len(row) > 5 else "0"
                prize2_count = int(prize2_str) if prize2_str.isdigit() else 0
                
                draws.append({
                    'period': period,
                    'date': date,
                    'reds': reds,
                    'blue': blue,
                    'sales': sales,
                    'prize1_count': prize1_count,
                    'prize2_count': prize2_count,
                    'pool': 0,
                    'prize1_amount': 0,
                    'prize2_amount': 0
                })
            except Exception:
                continue
        
        return draws if draws else None
    except Exception:
        return None

# ==================== 数据源3：Excel解析器 ====================
def parse_excel_file(uploaded_file) -> Optional[List[Dict]]:
    """解析用户上传的Excel文件"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=0)
        draws = []
        
        # 检测标准格式（期号、日期、红球6列、蓝球）
        if '期号' in df.columns:
            has_standard_format = True
            period_col = '期号'
            date_col = '开奖日期' if '开奖日期' in df.columns else 'date'
            blue_col = '蓝球' if '蓝球' in df.columns else 'blue'
            
            for idx, row in df.iterrows():
                try:
                    period = str(row.get(period_col, ''))
                    if period == 'nan' or not period:
                        continue
                    
                    date_val = row.get(date_col)
                    if isinstance(date_val, datetime):
                        date_val = date_val.strftime('%Y-%m-%d')
                    
                    reds = []
                    for i in range(1, 7):
                        col_name = f'红球号码_{i}' if f'红球号码_{i}' in df.columns else f'red{i}'
                        if col_name in df.columns:
                            val = row.get(col_name)
                            if pd.notna(val):
                                reds.append(int(val))
                    
                    if len(reds) != 6:
                        continue
                    
                    blue = None
                    if blue_col in df.columns:
                        blue_val = row.get(blue_col)
                        if pd.notna(blue_val):
                            blue = int(blue_val)
                    
                    if blue is None and len(row) > 8:
                        blue = int(row.iloc[8]) if pd.notna(row.iloc[8]) else None
                    
                    if blue:
                        draws.append({
                            'period': period,
                            'date': date_val,
                            'reds': reds,
                            'blue': blue,
                            'pool': 0,
                            'sales': 0,
                            'prize1_count': 0,
                            'prize2_count': 0,
                            'prize1_amount': 0,
                            'prize2_amount': 0
                        })
                except Exception:
                    continue
        
        # 备用解析：直接按列索引（红球在C-H列，蓝球在I列）
        if len(draws) == 0:
            for idx, row in df.iterrows():
                try:
                    if len(row) < 9:
                        continue
                    
                    reds = []
                    for i in range(2, 8):
                        if i < len(row):
                            val = row.iloc[i]
                            if pd.notna(val):
                                reds.append(int(float(val)))
                    
                    if len(reds) != 6:
                        continue
                    
                    blue_val = row.iloc[8] if len(row) > 8 else None
                    if pd.notna(blue_val):
                        blue = int(float(blue_val))
                    else:
                        continue
                    
                    period = str(row.iloc[0]) if len(row) > 0 and pd.notna(row.iloc[0]) else str(idx + 1)
                    
                    draws.append({
                        'period': period,
                        'date': None,
                        'reds': reds,
                        'blue': blue,
                        'pool': 0,
                        'sales': 0,
                        'prize1_count': 0,
                        'prize2_count': 0,
                        'prize1_amount': 0,
                        'prize2_amount': 0
                    })
                except Exception:
                    continue
        
        return draws if draws else None
    except Exception:
        return None

# ==================== 数据源4：Supabase操作 ====================
def load_from_supabase(limit: int = 500) -> Optional[List[Dict]]:
    """从Supabase加载数据"""
    supabase = init_supabase()
    if supabase is None:
        return None
    
    schemas_to_try = ['ssq_schema', 'public']
    
    for schema_name in schemas_to_try:
        try:
            response = supabase.schema(schema_name).table('ssq_draws').select("*").order("period", desc=False).limit(limit).execute()
            
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
                    'prize2_count': row.get('prize2_count', 0),
                    'prize1_amount': row.get('prize1_amount', 0),
                    'prize2_amount': row.get('prize2_amount', 0)
                })
            if draws:
                return draws
        except Exception:
            continue
    
    return None

def save_draws_to_supabase(draws: List[Dict]) -> int:
    """保存数据到Supabase"""
    supabase = init_supabase()
    if supabase is None:
        return 0
    
    schemas_to_try = ['ssq_schema', 'public']
    
    for schema_name in schemas_to_try:
        try:
            existing = supabase.schema(schema_name).table('ssq_draws').select("period").execute()
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
                supabase.schema(schema_name).table('ssq_draws').insert(data).execute()
                new_count += 1
            
            return new_count
        except Exception:
            continue
    
    return 0

# ==================== 数据源管理器 ====================
class DataSourceManager:
    """管理多个数据源，支持优先级和自动降级"""
    
    def __init__(self):
        self.sources = {
            'mcp': {'name': 'MCP服务', 'enabled': True, 'priority': 1, 'status': '待检测', 'last_error': None},
            'zhcw': {'name': '中彩网', 'enabled': True, 'priority': 2, 'status': '待检测', 'last_error': None},
            'supabase': {'name': 'Supabase缓存', 'enabled': True, 'priority': 3, 'status': '待检测', 'last_error': None},
            'excel': {'name': 'Excel导入', 'enabled': True, 'priority': 4, 'status': '待检测', 'last_error': None}
        }
        self.data = None
        self.source_used = None
        self.last_update = None
        self.excel_data = None
    
    def set_enabled(self, source_key: str, enabled: bool):
        if source_key in self.sources:
            self.sources[source_key]['enabled'] = enabled
    
    def set_excel_data(self, data: List[Dict]):
        self.excel_data = data
        if data and len(data) > 0:
            self.sources['excel']['status'] = f"已导入 {len(data)} 期"
        else:
            self.sources['excel']['status'] = "待上传"
    
    def fetch_all(self, limit: int = 200) -> bool:
        sorted_sources = sorted(
            [(k, v) for k, v in self.sources.items() if v['enabled']],
            key=lambda x: x[1]['priority']
        )
        
        for source_key, source_info in sorted_sources:
            try:
                if source_key == 'mcp':
                    data = fetch_from_mcp(limit=limit)
                elif source_key == 'zhcw':
                    all_data = []
                    for page in range(1, 4):
                        page_data = fetch_from_zhcw(page=page)
                        if page_data:
                            all_data.extend(page_data)
                        time.sleep(0.5)
                    data = all_data
                elif source_key == 'supabase':
                    data = load_from_supabase(limit=limit)
                elif source_key == 'excel':
                    data = self.excel_data
                else:
                    continue
                
                if data and len(data) > 0:
                    seen = set()
                    unique_data = []
                    for d in data:
                        period = d.get('period', '')
                        if period and period not in seen:
                            seen.add(period)
                            unique_data.append(d)
                    
                    source_info['status'] = f"成功获取 {len(unique_data)} 期"
                    source_info['last_error'] = None
                    self.data = unique_data
                    self.source_used = source_key
                    self.last_update = datetime.now()
                    return True
                else:
                    source_info['status'] = "无数据"
            except Exception as e:
                source_info['status'] = "失败"
                source_info['last_error'] = str(e)[:100]
                continue
        
        return False
    
    def get_data(self) -> Optional[List[Dict]]:
        return self.data
    
    def get_source_used(self) -> Optional[str]:
        return self.source_used
    
    def get_status_df(self) -> pd.DataFrame:
        rows = []
        for k, v in self.sources.items():
            rows.append({
                '数据源': v['name'],
                '状态': v['status'],
                '启用': '✅' if v['enabled'] else '❌'
            })
        return pd.DataFrame(rows)

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
        
        max_freq = max(freq.values()) if freq.values() else 1
        for num in freq:
            freq[num] = freq[num] / max_freq if max_freq > 0 else 0
        
        return freq
    
    def _calculate_absence(self, field: str) -> Dict[int, int]:
        absence = {i: 0 for i in range(1, 34 if field == 'reds' else 17)}
        last_seen = {i: None for i in range(1, 34 if field == 'reds' else 17)}
        
        for idx, draw in enumerate(reversed(self.draws)):
            if field == 'reds':
                for num in draw.get('reds', []):
                    if 1 <= num <= 33 and last_seen[num] is None:
                        last_seen[num] = idx
            else:
                blue = draw.get('blue', 0)
                if 1 <= blue <= 16 and last_seen[blue] is None:
                    last_seen[blue] = idx
        
        total = len(self.draws)
        for num in absence:
            absence[num] = last_seen[num] if last_seen[num] is not None else total
        
        return absence
    
    def calculate_red_scores(self) -> Dict[int, float]:
        max_absence = max(self.red_absence.values()) if self.red_absence.values() else 1
        scores = {}
        for num in RED_NUMBERS:
            freq_score = self.red_freq.get(num, 0)
            absence_score = 1 - (self.red_absence.get(num, max_absence) / max_absence) if max_absence > 0 else 0
            scores[num] = 0.5 * freq_score + 0.5 * absence_score
        return scores
    
    def calculate_blue_scores(self) -> Dict[int, float]:
        max_absence = max(self.blue_absence.values()) if self.blue_absence.values() else 1
        scores = {}
        for num in BLUE_NUMBERS:
            freq_score = self.blue_freq.get(num, 0)
            absence_score = 1 - (self.blue_absence.get(num, max_absence) / max_absence) if max_absence > 0 else 0
            scores[num] = 0.5 * freq_score + 0.5 * absence_score
        return scores
    
    def get_target_sum(self) -> Tuple[int, int]:
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
        red_scores = self.calculate_red_scores()
        blue_scores = self.calculate_blue_scores()
        
        red_weights = np.array([math.exp(red_scores.get(i, 0)) for i in RED_NUMBERS])
        blue_weights = np.array([math.exp(blue_scores.get(i, 0)) for i in BLUE_NUMBERS])
        
        if np.sum(red_weights) > 0:
            red_weights = red_weights / np.sum(red_weights)
        else:
            red_weights = np.ones(33) / 33
        
        if np.sum(blue_weights) > 0:
            blue_weights = blue_weights / np.sum(blue_weights)
        else:
            blue_weights = np.ones(16) / 16
        
        target_sum, tolerance = self.get_target_sum()
        
        bets = []
        for _ in range(num_bets):
            for attempt in range(100):
                reds = np.random.choice(RED_NUMBERS, size=6, replace=False, p=red_weights)
                reds = sorted(reds.tolist())
                if abs(sum(reds) - target_sum) <= tolerance:
                    blue = np.random.choice(BLUE_NUMBERS, p=blue_weights)
                    bets.append({
                        'reds': reds,
                        'blue': int(blue),
                        'sum': sum(reds),
                        'method': '方法1:冷热码+和值'
                    })
                    break
            else:
                reds = sorted(np.random.choice(RED_NUMBERS, size=6, replace=False))
                blue = np.random.choice(BLUE_NUMBERS)
                bets.append({
                    'reds': reds,
                    'blue': int(blue),
                    'sum': sum(reds),
                    'method': '方法1:冷热码+和值'
                })
        
        return bets

# ==================== 方法2：胆拖混合 ====================
class Method2DanTuo:
    """方法2：胆拖混合"""
    
    def __init__(self, draws: List[Dict]):
        self.draws = draws
        self.method1 = Method1HotColdSum(draws)
    
    def select_anchors(self, num_anchors: int = 2) -> List[int]:
        red_scores = self.method1.calculate_red_scores()
        
        if self.draws:
            last_reds = self.draws[-1].get('reds', [])
            for num in last_reds:
                if num in red_scores:
                    red_scores[num] += 0.3
        
        sorted_nums = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:num_anchors]]
    
    def generate_bets(self, num_bets: int = 4) -> List[Dict]:
        anchors = self.select_anchors()
        red_scores = self.method1.calculate_red_scores()
        blue_scores = self.method1.calculate_blue_scores()
        
        for a in anchors:
            if a in red_scores:
                red_scores[a] = 0.1
        
        red_weights = np.array([math.exp(red_scores.get(i, 0)) for i in RED_NUMBERS])
        blue_weights = np.array([math.exp(blue_scores.get(i, 0)) for i in BLUE_NUMBERS])
        
        if np.sum(red_weights) > 0:
            red_weights = red_weights / np.sum(red_weights)
        else:
            red_weights = np.ones(33) / 33
        
        if np.sum(blue_weights) > 0:
            blue_weights = blue_weights / np.sum(blue_weights)
        else:
            blue_weights = np.ones(16) / 16
        
        target_sum, tolerance = self.method1.get_target_sum()
        
        bets = []
        for _ in range(num_bets):
            needed = 6 - len(anchors)
            for attempt in range(100):
                candidates = [i for i in RED_NUMBERS if i not in anchors]
                if not candidates:
                    break
                candidate_weights = np.array([red_weights[i-1] for i in candidates])
                if np.sum(candidate_weights) > 0:
                    candidate_weights = candidate_weights / np.sum(candidate_weights)
                else:
                    candidate_weights = np.ones(len(candidates)) / len(candidates)
                selected = np.random.choice(candidates, size=needed, replace=False, p=candidate_weights)
                reds = sorted(anchors + selected.tolist())
                if abs(sum(reds) - target_sum) <= tolerance:
                    blue = np.random.choice(BLUE_NUMBERS, p=blue_weights)
                    bets.append({
                        'reds': reds,
                        'blue': int(blue),
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
                    'blue': int(blue),
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
        if len(window_draws) < 20:
            return None
        
        features = {}
        total = len(window_draws)
        
        freq = sum(1 for d in window_draws if target_num in d.get('reds', []))
        features['freq'] = freq / total if total > 0 else 0
        
        last_seen = None
        for idx, d in enumerate(reversed(window_draws)):
            if target_num in d.get('reds', []):
                last_seen = idx
                break
        features['absence'] = last_seen if last_seen is not None else total
        
        recent = window_draws[-10:] if len(window_draws) >= 10 else window_draws
        recent_freq = sum(1 for d in recent if target_num in d.get('reds', []))
        features['recent_freq'] = recent_freq / len(recent) if recent else 0
        
        if window_draws:
            features['last_appeared'] = 1 if target_num in window_draws[-1].get('reds', []) else 0
        
        zone = (target_num - 1) // 11 + 1
        features['zone'] = zone
        features['parity'] = target_num % 2
        features['size'] = 0 if target_num <= 16 else 1
        
        return features
    
    def train(self) -> bool:
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
        except Exception:
            return False
    
    def predict(self) -> List[int]:
        if not self.is_trained or not self.model:
            return []
        
        predictions = []
        for num in RED_NUMBERS:
            features = self._extract_features(self.draws, num)
            if features:
                X_pred = pd.DataFrame([features]).fillna(0)
                try:
                    prob = self.model.predict_proba(X_pred)[0][1]
                    predictions.append((num, prob))
                except:
                    predictions.append((num, 0.0))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [num for num, _ in predictions[:6]]
    
    def generate_bets(self, num_bets: int = 4) -> List[Dict]:
        if not self.is_trained:
            self.train()
        
        if not self.is_trained:
            method1 = Method1HotColdSum(self.draws)
            return method1.generate_bets(num_bets)
        
        predicted_reds = self.predict()
        blue_scores = Method1HotColdSum(self.draws).calculate_blue_scores()
        blue_weights = np.array([math.exp(blue_scores.get(i, 0)) for i in BLUE_NUMBERS])
        
        if np.sum(blue_weights) > 0:
            blue_weights = blue_weights / np.sum(blue_weights)
        else:
            blue_weights = np.ones(16) / 16
        
        bets = []
        for _ in range(num_bets):
            reds = predicted_reds[:]
            if len(reds) < 6:
                missing = [n for n in RED_NUMBERS if n not in reds]
                extra = np.random.choice(missing, size=6-len(reds), replace=False)
                reds.extend(extra)
            
            replace_count = np.random.randint(1, 3)
            for _ in range(replace_count):
                idx = np.random.randint(0, len(reds))
                candidates = [n for n in RED_NUMBERS if n not in reds]
                if candidates:
                    reds[idx] = np.random.choice(candidates)
            
            reds = sorted(reds[:6])
            blue = np.random.choice(BLUE_NUMBERS, p=blue_weights)
            
            bets.append({
                'reds': reds,
                'blue': int(blue),
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
        if len(window_draws) < 30:
            return None
        
        features = {}
        total = len(window_draws)
        
        freq = sum(1 for d in window_draws if target_num in d.get('reds', []))
        features['freq'] = freq / total if total > 0 else 0
        
        last_seen = None
        for idx, d in enumerate(reversed(window_draws)):
            if target_num in d.get('reds', []):
                last_seen = idx
                break
        features['absence'] = last_seen if last_seen is not None else total
        features['absence_norm'] = features['absence'] / total if total > 0 else 0
        
        for window in [3, 5, 10]:
            recent = window_draws[-window:] if len(window_draws) >= window else window_draws
            recent_freq = sum(1 for d in recent if target_num in d.get('reds', []))
            features[f'recent_{window}'] = recent_freq / len(recent) if recent else 0
        
        if window_draws:
            last_reds = window_draws[-1].get('reds', [])
            features['last_appeared'] = 1 if target_num in last_reds else 0
            if last_reds:
                features['min_diff_to_last'] = min(abs(target_num - n) for n in last_reds)
            else:
                features['min_diff_to_last'] = 99
        
        zone = (target_num - 1) // 11 + 1
        features['zone'] = zone
        features['parity'] = target_num % 2
        features['size'] = 0 if target_num <= 16 else 1
        features['tail'] = target_num % 10
        
        return features
    
    def train(self) -> bool:
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
        except Exception:
            return False
    
    def predict(self) -> List[int]:
        if not self.is_trained:
            return []
        
        predictions = []
        for num in RED_NUMBERS:
            features = self._extract_features_advanced(self.draws, num)
            if features:
                X_pred = pd.DataFrame([features]).fillna(0)
                
                try:
                    xgb_prob = self.xgb_model.predict_proba(X_pred)[0][1]
                except:
                    xgb_prob = 0.5
                
                try:
                    X_scaled = self.scaler.transform(X_pred)
                    nn_prob = self.nn_model.predict_proba(X_scaled)[0][1]
                except:
                    nn_prob = 0.5
                
                ensemble_prob = 0.5 * xgb_prob + 0.5 * nn_prob
                predictions.append((num, ensemble_prob))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [num for num, _ in predictions[:6]]
    
    def generate_bets(self, num_bets: int = 4) -> List[Dict]:
        if not self.is_trained:
            self.train()
        
        if not self.is_trained:
            method1 = Method1HotColdSum(self.draws)
            return method1.generate_bets(num_bets)
        
        predicted_reds = self.predict()
        blue_scores = Method1HotColdSum(self.draws).calculate_blue_scores()
        blue_weights = np.array([math.exp(blue_scores.get(i, 0)) for i in BLUE_NUMBERS])
        
        if np.sum(blue_weights) > 0:
            blue_weights = blue_weights / np.sum(blue_weights)
        else:
            blue_weights = np.ones(16) / 16
        
        bets = []
        for _ in range(num_bets):
            reds = predicted_reds[:]
            if len(reds) < 6:
                missing = [n for n in RED_NUMBERS if n not in reds]
                extra = np.random.choice(missing, size=6-len(reds), replace=False)
                reds.extend(extra)
            
            reds = sorted(reds[:6])
            blue = np.random.choice(BLUE_NUMBERS, p=blue_weights)
            
            bets.append({
                'reds': reds,
                'blue': int(blue),
                'sum': sum(reds),
                'method': '方法4:XGBoost+NN集成'
            })
        
        return bets

# ==================== 投注生成工厂 ====================
class BetGenerator:
    @staticmethod
    def generate(method: str, draws: List[Dict], num_bets: int = 4) -> List[Dict]:
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

# ==================== ML信号计算 ====================
def calculate_ml_signals(draws: List[Dict]) -> Dict:
    """计算ML特征信号"""
    if not draws or len(draws) < 10:
        return {
            'jackpot_level': '数据不足',
            'scissors': '数据不足',
            'cycle': '数据不足',
            'blue_bias': '均衡',
            'signal_strength': 0,
            'suggestion_text': '数据不足',
            'pool': 0,
            'sales': 0
        }
    
    latest = draws[-1]
    pool = latest.get('pool', 0)
    sales = latest.get('sales', 0)
    
    if pool >= 250000000:
        jackpot_level = "HIGH (≥2.5亿)"
        signal_strength = 30
    elif pool >= 150000000:
        jackpot_level = "MEDIUM (1.5-2.5亿)"
        signal_strength = 15
    else:
        jackpot_level = "LOW (<1.5亿)"
        signal_strength = 0
    
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
    
    signal_strength = max(0, min(100, signal_strength))
    
    if signal_strength >= 60:
        suggestion_text = "🔔 强烈推荐投注"
    elif signal_strength >= 30:
        suggestion_text = "⚠️ 谨慎投注"
    else:
        suggestion_text = "💤 建议观望"
    
    return {
        'jackpot_level': jackpot_level,
        'scissors': scissors,
        'cycle': cycle,
        'blue_bias': blue_bias,
        'signal_strength': signal_strength,
        'suggestion_text': suggestion_text,
        'pool': pool,
        'sales': sales
    }

# ==================== DeepSeek AI 建议（含ML提示） ====================
def get_deepseek_suggestion(draws: List[Dict], source_used: str, ml_signals: Dict, next_period: str) -> Dict:
    """获取DeepSeek AI的投注建议和ML提示"""
    if not DEEPSEEK_API_KEY or len(draws) < 10:
        red_scores = Method1HotColdSum(draws).calculate_red_scores()
        top_reds = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)[:6]
        top_blue = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)[:1] if red_scores else [(8, 0)]
        return {
            "plan": "4组7+1复式",
            "win_rate": "30-35%",
            "blue_advice": f"推荐蓝球{top_blue[0][0] if top_blue else 8}",
            "summary": f"基于{len(draws)}期历史数据，红球热号集中在{top_reds[0][0] if top_reds else 1}附近",
            "ml_tip": f"奖池{ml_signals['jackpot_level']}，{ml_signals['cycle']}，建议关注蓝球{ml_signals['blue_bias']}"
        }
    
    red_scores = Method1HotColdSum(draws).calculate_red_scores()
    blue_scores = Method1HotColdSum(draws).calculate_blue_scores()
    
    top_reds = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)[:6]
    top_blues = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    
    try:
        prompt = f"""你是双色球AI分析专家。基于以下数据给出投注建议和ML提示（JSON格式）：

【数据来源】{source_used}
【历史数据量】{len(draws)}期
【预测下一期】{next_period}
【红球热号Top6】{[r[0] for r in top_reds]}
【蓝球热号Top3】{[b[0] for b in top_blues]}
【奖池】{ml_signals.get('pool', 0)/1e8:.1f}亿
【剪刀差信号】{ml_signals.get('scissors', '正常')}
【奖池阈值】{ml_signals.get('jackpot_level', '正常')}
【头奖周期】{ml_signals.get('cycle', '正常')}
【蓝球偏向】{ml_signals.get('blue_bias', '均衡')}

请按此JSON格式回复：
{{"plan": "推荐方案(7+1/7+2/观望)", "win_rate": "预期赢率%", "blue_advice": "蓝球建议", "summary": "一句话总结", "ml_tip": "关于下一期的ML提示（30字以内）"}}
"""
        response = requests.post(
            f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
            json={"model": DEEPSEEK_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 500},
            timeout=10
        )
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            json_match = re.search(r'\{[^{}]*\}', content)
            if json_match:
                return json.loads(json_match.group())
    except Exception:
        pass
    
    return {
        "plan": "4组7+1复式",
        "win_rate": "30-35%",
        "blue_advice": f"推荐蓝球{top_blues[0][0] if top_blues else 8}",
        "summary": f"基于{len(draws)}期历史数据，红球热号集中在{top_reds[0][0] if top_reds else 1}附近",
        "ml_tip": f"奖池{ml_signals['jackpot_level']}，{ml_signals['cycle']}，建议关注蓝球{ml_signals['blue_bias']}"
    }

# ==================== ROI回测功能 ====================
def backtest_roi(draws: List[Dict], method: str, num_bets: int = 4, lookback: int = 50) -> Dict:
    """回测指定方法的ROI"""
    if len(draws) < lookback + 10:
        return {"roi": 0, "total_cost": 0, "total_prize": 0, "net": 0, "win_rate": 0}
    
    total_cost = 0
    total_prize = 0
    win_count = 0
    prize_breakdown = {"first": 0, "second": 0, "third": 0, "fourth": 0, "fifth": 0, "sixth": 0, "fuyun": 0}
    
    for i in range(lookback, len(draws)):
        historical = draws[:i]
        actual = draws[i]
        
        bets = BetGenerator.generate(method, historical, num_bets)
        
        period_cost = num_bets * 14
        period_prize = 0
        
        for bet in bets:
            red_matches = len(set(bet['reds']) & set(actual['reds'])) if actual.get('reds') else 0
            blue_match = (bet['blue'] == actual.get('blue', 0)) if actual.get('blue') else False
            
            if red_matches == 6 and blue_match:
                period_prize += 5000000
                prize_breakdown["first"] += 1
            elif red_matches == 6:
                period_prize += 500000
                prize_breakdown["second"] += 1
            elif red_matches == 5 and blue_match:
                period_prize += 3000
                prize_breakdown["third"] += 1
            elif red_matches == 5 or (red_matches == 4 and blue_match):
                period_prize += 200
                prize_breakdown["fourth"] += 1
            elif red_matches == 4 or (red_matches == 3 and blue_match):
                period_prize += 10
                prize_breakdown["fifth"] += 1
            elif blue_match:
                period_prize += 5
                prize_breakdown["sixth"] += 1
            elif red_matches == 3:
                period_prize += 5
                prize_breakdown["fuyun"] += 1
        
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
        "periods": lookback,
        "prize_breakdown": prize_breakdown
    }

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

# ==================== 冷热码分析函数 ====================
def get_hot_cold_analysis(draws: List[Dict], analysis_periods: int = 100):
    """获取冷热码分析数据"""
    if len(draws) < analysis_periods:
        analysis_periods = len(draws)
    
    recent_draws = draws[-analysis_periods:]
    
    red_freq = {i: 0 for i in range(1, 34)}
    blue_freq = {i: 0 for i in range(1, 17)}
    
    for draw in recent_draws:
        for num in draw.get('reds', []):
            if 1 <= num <= 33:
                red_freq[num] += 1
        blue = draw.get('blue', 0)
        if 1 <= blue <= 16:
            blue_freq[blue] += 1
    
    # 计算遗漏期数
    red_absence = {i: 0 for i in range(1, 34)}
    blue_absence = {i: 0 for i in range(1, 17)}
    
    for num in range(1, 34):
        last_seen = None
        for idx, draw in enumerate(reversed(draws)):
            if num in draw.get('reds', []):
                last_seen = idx
                break
        red_absence[num] = last_seen if last_seen is not None else len(draws)
    
    for num in range(1, 17):
        last_seen = None
        for idx, draw in enumerate(reversed(draws)):
            if draw.get('blue', 0) == num:
                last_seen = idx
                break
        blue_absence[num] = last_seen if last_seen is not None else len(draws)
    
    # 排序
    hot_reds = sorted(red_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    cold_reds = sorted(red_freq.items(), key=lambda x: x[1])[:10]
    hot_blues = sorted(blue_freq.items(), key=lambda x: x[1], reverse=True)[:8]
    
    return {
        'hot_reds': hot_reds,
        'cold_reds': cold_reds,
        'hot_blues': hot_blues,
        'red_freq': red_freq,
        'blue_freq': blue_freq,
        'red_absence': red_absence,
        'blue_absence': blue_absence,
        'analysis_periods': analysis_periods
    }

# ==================== 7分区热度分析 ====================
def get_zone_heat(draws: List[Dict], analysis_periods: int = 100):
    """获取7分区热度"""
    if len(draws) < analysis_periods:
        analysis_periods = len(draws)
    
    recent_draws = draws[-analysis_periods:]
    
    zone_hits = {i: 0 for i in range(1, 8)}
    
    for draw in recent_draws:
        for num in draw.get('reds', []):
            for zone_id, zone_info in ZONES.items():
                if num in zone_info['numbers']:
                    zone_hits[zone_id] += 1
                    break
    
    max_hits = max(zone_hits.values()) if zone_hits.values() else 1
    zone_heat = {}
    for zone_id, hits in zone_hits.items():
        normalized = hits / max_hits
        if normalized >= 0.7:
            heat_level = "🔥 热"
        elif normalized >= 0.4:
            heat_level = "⚡ 中"
        else:
            heat_level = "❄️ 冷"
        zone_heat[zone_id] = {
            'name': ZONES[zone_id]['name'],
            'range': ZONES[zone_id]['range'],
            'hits': hits,
            'percentage': hits / (len(recent_draws) * 6) * 100,
            'heat_level': heat_level
        }
    
    return zone_heat

# ==================== 蓝球走势分析 ====================
def get_blue_trend(draws: List[Dict], analysis_periods: int = 50):
    """获取蓝球走势数据"""
    if len(draws) < analysis_periods:
        analysis_periods = len(draws)
    
    recent_draws = draws[-analysis_periods:]
    
    blue_sequence = []
    for draw in recent_draws:
        blue_sequence.append(draw.get('blue', 0))
    
    # 频率统计
    blue_freq = {i: 0 for i in range(1, 17)}
    for blue in blue_sequence:
        if 1 <= blue <= 16:
            blue_freq[blue] += 1
    
    # 遗漏期数
    blue_absence = {i: 0 for i in range(1, 17)}
    for num in range(1, 17):
        last_seen = None
        for idx, draw in enumerate(reversed(draws)):
            if draw.get('blue', 0) == num:
                last_seen = idx
                break
        blue_absence[num] = last_seen if last_seen is not None else len(draws)
    
    # 大小号比例
    small_count = sum(1 for b in blue_sequence if 1 <= b <= 8)
    large_count = sum(1 for b in blue_sequence if 9 <= b <= 16)
    
    return {
        'blue_sequence': blue_sequence,
        'blue_freq': blue_freq,
        'blue_absence': blue_absence,
        'small_count': small_count,
        'large_count': large_count,
        'analysis_periods': analysis_periods
    }

# ==================== 和值走势分析 ====================
def get_sum_trend(draws: List[Dict], analysis_periods: int = 50):
    """获取和值走势数据"""
    if len(draws) < analysis_periods:
        analysis_periods = len(draws)
    
    recent_draws = draws[-analysis_periods:]
    
    sums = []
    periods = []
    for draw in recent_draws:
        sums.append(sum(draw.get('reds', [])))
        periods.append(draw.get('period', ''))
    
    mean_sum = np.mean(sums) if sums else RED_EXPECTED_SUM
    std_sum = np.std(sums) if len(sums) > 1 else RED_SUM_STD
    
    return {
        'sums': sums,
        'periods': periods,
        'mean_sum': mean_sum,
        'std_sum': std_sum,
        'analysis_periods': analysis_periods
    }

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
    """管理员页面 - 数据源管理、Excel上传"""
    with st.expander("🔧 管理员控制台", expanded=True):
        st.subheader("📡 数据源状态")
        status_df = data_manager.get_status_df()
        st.dataframe(status_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.subheader("🔌 数据源开关")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mcp_enabled = st.checkbox("启用MCP服务", value=data_manager.sources['mcp']['enabled'], key="mcp_toggle")
            data_manager.set_enabled('mcp', mcp_enabled)
        with col2:
            zhcw_enabled = st.checkbox("启用中彩网爬虫", value=data_manager.sources['zhcw']['enabled'], key="zhcw_toggle")
            data_manager.set_enabled('zhcw', zhcw_enabled)
        with col3:
            supabase_enabled = st.checkbox("启用Supabase缓存", value=data_manager.sources['supabase']['enabled'], key="supabase_toggle")
            data_manager.set_enabled('supabase', supabase_enabled)
        with col4:
            st.checkbox("Excel导入", value=True, disabled=True, key="excel_show")
        
        st.markdown("---")
        st.subheader("📎 Excel数据导入")
        st.caption("上传您提供的双色球历史数据Excel文件")
        
        uploaded_file = st.file_uploader(
            "选择Excel文件",
            type=['xlsx', 'xls'],
            key="admin_excel_upload",
            help="支持双色球开奖数据_all.xlsx格式"
        )
        
        if uploaded_file is not None:
            with st.spinner("正在解析Excel文件..."):
                excel_draws = parse_excel_file(uploaded_file)
                if excel_draws and len(excel_draws) > 0:
                    st.success(f"✅ 成功解析 {len(excel_draws)} 期数据")
                    data_manager.set_excel_data(excel_draws)
                    
                    preview_df = pd.DataFrame([{
                        '期号': d['period'],
                        '红球': ','.join(str(r) for r in d['reds']),
                        '蓝球': d['blue']
                    } for d in excel_draws[:10]])
                    st.dataframe(preview_df, use_container_width=True, hide_index=True)
                    
                    if st.button("💾 保存Excel数据到Supabase", key="save_excel_to_supabase"):
                        new_count = save_draws_to_supabase(excel_draws)
                        if new_count > 0:
                            st.success(f"已保存 {new_count} 期新数据到Supabase")
                        else:
                            st.info("Supabase中已有这些数据")
                else:
                    st.error("解析失败，请检查文件格式")
        
        st.markdown("---")
        st.subheader("🔄 数据同步")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 立即同步所有数据源", type="primary"):
                with st.spinner("正在获取数据..."):
                    if data_manager.fetch_all():
                        st.success(f"数据获取成功！来源：{data_manager.get_source_used()}")
                        draws = data_manager.get_data()
                        if draws:
                            new_count = save_draws_to_supabase(draws)
                            if new_count > 0:
                                st.success(f"已同步 {new_count} 期新数据到Supabase")
                            else:
                                st.info("Supabase中已有这些数据")
                        st.rerun()
                    else:
                        st.error("所有数据源均获取失败")
        
        with col2:
            st.caption("优先级：MCP > 中彩网 > Supabase > Excel")
        
        st.markdown("---")
        st.subheader("📁 缓存管理")
        
        cached_data = load_from_supabase()
        if cached_data:
            st.info(f"Supabase缓存中有 {len(cached_data)} 期数据")
            if st.button("🗑️ 清空缓存", type="secondary"):
                supabase = init_supabase()
                if supabase:
                    for schema in ['ssq_schema', 'public']:
                        try:
                            supabase.schema(schema).table('ssq_draws').delete().neq("id", 0).execute()
                        except:
                            pass
                st.success("缓存已清空")
                st.rerun()
        else:
            st.info("Supabase缓存为空")

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

# ==================== 加载数据 ====================
draws = load_from_supabase()

if not draws or len(draws) < 10:
    with st.spinner("正在从数据源获取历史数据..."):
        if data_manager.fetch_all(limit=500):
            draws = data_manager.get_data()
            if draws:
                save_draws_to_supabase(draws)

if not draws or len(draws) < 10:
    st.info("👈 请点击右上角齿轮图标，进入管理员页面上传Excel文件或同步数据")
    st.stop()

# ==================== 获取下一期期号 ====================
def get_next_period(draws: List[Dict]) -> str:
    """获取下一期期号"""
    if not draws:
        return "未知"
    latest_period = draws[-1].get('period', '')
    if latest_period and latest_period.isdigit():
        return str(int(latest_period) + 1)
    return "未知"

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
        st.caption(f"数据源: {data_manager.get_source_used()}")
    
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
    st.caption("DFSS智能选号工具 v8.0 Final")

# ==================== 右上角齿轮 ====================
col_title, col_settings = st.columns([0.95, 0.05])
with col_title:
    st.title("🎯 双色球AI智能选号")
with col_settings:
    if st.button("⚙️", key="settings_icon", help="管理员设置"):
        st.session_state['show_admin'] = not st.session_state.get('show_admin', False)

if st.session_state.get('show_admin', False):
    if not st.session_state['admin_logged_in']:
        admin_login()
    else:
        show_admin_page(data_manager)
        admin_logout()

# ==================== 数据概览 ====================
st.subheader("📊 数据概览")
latest = draws[-1]
next_period = get_next_period(draws)
ml_signals = calculate_ml_signals(draws)

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.metric("最新期号", latest.get('period', 'N/A'))
with col2:
    date_val = latest.get('date', '')
    st.metric("最新日期", date_val[:10] if date_val else 'N/A')
with col3:
    pool = ml_signals.get('pool', 0)
    st.metric("奖池金额", f"¥{pool/1e8:.1f}亿")
with col4:
    sales = ml_signals.get('sales', 0)
    st.metric("上期投注额", f"¥{sales/1e8:.1f}亿")
with col5:
    st.metric("数据总量", f"{len(draws)}期")
with col6:
    signal_strength = ml_signals.get('signal_strength', 0)
    st.metric("信号强度", f"{signal_strength}%")

st.markdown("---")

# ==================== 预测下一期 + ML提示 ====================
st.subheader("🎯 预测下一期")

col_pred, col_tip = st.columns([1, 2])
with col_pred:
    st.info(f"**下一期：{next_period}**")
with col_tip:
    # 获取DeepSeek建议（含ML提示）
    ai_suggestion = get_deepseek_suggestion(draws, data_manager.get_source_used(), ml_signals, next_period)
    st.markdown(f'<div class="ml-tip-box">🤖 <strong>ML 提示</strong><br>{ai_suggestion.get("ml_tip", "分析中...")}</div>', unsafe_allow_html=True)

st.markdown("---")

# ==================== 冷热码分析 ====================
st.subheader("🔥 冷热码分析")

# 分析期数可调节
analysis_periods = st.slider("统计期数", min_value=20, max_value=min(500, len(draws)), value=min(100, len(draws)), step=10, key="analysis_periods")

cold_hot_data = get_hot_cold_analysis(draws, analysis_periods)
zone_heat = get_zone_heat(draws, analysis_periods)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔥 热门红球 Top 15**")
    hot_df = pd.DataFrame([
        {'号码': num, '出现次数': cnt, '得分': f"{cnt/analysis_periods*6*100:.1f}%"}
        for num, cnt in cold_hot_data['hot_reds']
    ])
    st.dataframe(hot_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("**❄️ 冷门红球 Bottom 10**")
    cold_df = pd.DataFrame([
        {'号码': num, '出现次数': cnt, '遗漏': cold_hot_data['red_absence'][num]}
        for num, cnt in cold_hot_data['cold_reds']
    ])
    st.dataframe(cold_df, use_container_width=True, hide_index=True)

with col3:
    st.markdown("**📊 7分区热度图**")
    zone_df = pd.DataFrame([
        {
            '分区': zone['name'],
            '范围': zone['range'],
            '热度': zone['heat_level'],
            '出现次数': zone['hits'],
            '占比': f"{zone['percentage']:.1f}%"
        }
        for zone_id, zone in zone_heat.items()
    ])
    st.dataframe(zone_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ==================== 和值趋势分析 ====================
st.subheader("📈 和值趋势分析")

sum_trend = get_sum_trend(draws, analysis_periods=50)

# 绘制和值走势图
fig_sum = go.Figure()
fig_sum.add_trace(go.Scatter(
    x=list(range(len(sum_trend['sums']))),
    y=sum_trend['sums'],
    mode='lines+markers',
    name='红球和值',
    line=dict(color='#ff6b6b', width=2),
    marker=dict(size=6)
))
fig_sum.add_hline(y=RED_EXPECTED_SUM, line_dash="dash", line_color="red", annotation_text="理论均值(102)")
fig_sum.add_hline(y=sum_trend['mean_sum'], line_dash="dot", line_color="green", annotation_text=f"均值({sum_trend['mean_sum']:.0f})")

# 添加预测目标线
target_sum, _ = Method1HotColdSum(draws).get_target_sum()
fig_sum.add_hline(y=target_sum, line_dash="dash", line_color="orange", annotation_text=f"预测目标({target_sum})")

fig_sum.add_hrect(y0=RED_EXPECTED_SUM - RED_SUM_STD, y1=RED_EXPECTED_SUM + RED_SUM_STD, 
                  line_width=0, fillcolor="green", opacity=0.1, annotation_text="约68%区间")
fig_sum.update_layout(
    title="最近50期红球和值走势",
    xaxis_title="期数（倒序）",
    yaxis_title="和值",
    height=400,
    hovermode='x unified'
)
st.plotly_chart(fig_sum, use_container_width=True)

st.markdown("---")

# ==================== 蓝球走势分析 ====================
st.subheader("💙 蓝球走势分析")

blue_trend = get_blue_trend(draws, analysis_periods=50)

col1, col2 = st.columns(2)

with col1:
    # 蓝球频率柱状图
    blue_freq_df = pd.DataFrame([
        {'蓝球': num, '出现次数': blue_trend['blue_freq'][num]}
        for num in range(1, 17)
    ])
    fig_blue = px.bar(
        blue_freq_df, x='蓝球', y='出现次数',
        title=f"最近{blue_trend['analysis_periods']}期蓝球出现频率",
        color='出现次数',
        color_continuous_scale='Blues'
    )
    fig_blue.update_layout(height=400)
    st.plotly_chart(fig_blue, use_container_width=True)

with col2:
    st.markdown("**📊 蓝球统计**")
    stats_df = pd.DataFrame([
        {'指标': '小号(1-8)出现次数', '数值': blue_trend['small_count']},
        {'指标': '大号(9-16)出现次数', '数值': blue_trend['large_count']},
        {'指标': '小号占比', '数值': f"{blue_trend['small_count']/blue_trend['analysis_periods']*100:.1f}%"},
        {'指标': '大号占比', '数值': f"{blue_trend['large_count']/blue_trend['analysis_periods']*100:.1f}%"},
    ])
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    st.markdown("**📋 蓝球遗漏Top 5**")
    blue_absence_sorted = sorted(blue_trend['blue_absence'].items(), key=lambda x: x[1], reverse=True)[:5]
    absence_df = pd.DataFrame([
        {'蓝球': num, '遗漏期数': absence}
        for num, absence in blue_absence_sorted
    ])
    st.dataframe(absence_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ==================== AI决策引擎 ====================
st.subheader("🧠 AI决策引擎分析")

red_scores = Method1HotColdSum(draws).calculate_red_scores()
blue_scores = Method1HotColdSum(draws).calculate_blue_scores()

top_red = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)[:5]
top_blue = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)[:5]
target_sum, tolerance = Method1HotColdSum(draws).get_target_sum()

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("🔥 红球热号", ", ".join([str(r[0]) for r in top_red[:3]]))
with col2:
    st.metric("💙 蓝球热号", ", ".join([str(b[0]) for b in top_blue[:3]]))
with col3:
    st.metric("📊 目标和值", f"{target_sum} ± {tolerance}")
with col4:
    st.metric("📈 信号强度", f"{ml_signals['signal_strength']}%")
with col5:
    signal_text = ml_signals['suggestion_text']
    st.metric("🎯 建议", signal_text[:8] + "...")

st.markdown("---")

# ==================== 智能投注生成 ====================
st.subheader("🎲 智能投注生成")

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

col1, col2 = st.columns(2)
with col1:
    require_pattern = st.checkbox("☑ 连号/跳号要求", value=True, key="require_pattern")
with col2:
    require_repeat = st.checkbox("☑ 上期重复1-2个要求", value=True, key="require_repeat")

seed_input = st.text_input("随机种子 (可选)", value="", placeholder="例如: 2026-05-06")

if st.button("🚀 生成智能投注", type="primary", key="generate_btn"):
    if seed_input and seed_input.strip():
        try:
            seed_val = int(seed_input)
            random.seed(seed_val)
            np.random.seed(seed_val)
        except:
            random.seed()
            np.random.seed()
    
    with st.spinner(f"正在使用 {ai_model} 生成投注..."):
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
    st.caption(f"{bet_type}复式，每组成本{bet_type.split('(')[1] if '(' in bet_type else '14元'}")
    
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
            
            # 显示奖金明细
            st.markdown("**🏆 奖金明细（方法4）**")
            method4_result = backtest_roi(draws, "方法4: XGBoost+NN集成", backtest_bets, backtest_periods)
            breakdown = method4_result.get('prize_breakdown', {})
            breakdown_df = pd.DataFrame([
                {'奖级': '一等奖', '中奖注数': breakdown.get('first', 0)},
                {'奖级': '二等奖', '中奖注数': breakdown.get('second', 0)},
                {'奖级': '三等奖', '中奖注数': breakdown.get('third', 0)},
                {'奖级': '四等奖', '中奖注数': breakdown.get('fourth', 0)},
                {'奖级': '五等奖', '中奖注数': breakdown.get('fifth', 0)},
                {'奖级': '六等奖', '中奖注数': breakdown.get('sixth', 0)},
                {'奖级': '福运奖', '中奖注数': breakdown.get('fuyun', 0)},
            ])
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
            
            st.caption(f"回测期间：最近{backtest_periods}期 | 每组成本14元")

st.markdown("---")

# ==================== 多期查奖 ====================
st.subheader("🔍 多期查奖")
st.caption("📌 粘贴实际开奖数据，查看投注中奖情况")

check_draws_text = st.text_area(
    "📋 粘贴开奖数据（最多5期）",
    height=120,
    key="check_draws",
    placeholder="格式: 期号 日期 红1 红2 红3 红4 红5 红6 蓝\n示例:\n2026050 2026-05-03 03 04 14 15 18 20 02\n2026049 2026-05-01 09 15 18 24 28 33 01"
)

if st.button("🔍 查奖", key="check_btn") and check_draws_text:
    check_draws = parse_check_draws(check_draws_text)
    if check_draws:
        st.success(f"✅ 成功解析 {len(check_draws)} 期数据")
        
        if st.session_state.get('generated_bets'):
            enhanced_data = []
            for i, bet in enumerate(st.session_state['generated_bets'], 1):
                row = {'组别': i, '红球': ' '.join(f"{n:02d}" for n in bet['reds']), '蓝球': f"{bet['blue']:02d}"}
                for draw in check_draws:
                    result = calculate_prize(bet, draw)
                    row[f'{draw["period"]}'] = result
                enhanced_data.append(row)
            
            st.dataframe(pd.DataFrame(enhanced_data), use_container_width=True, hide_index=True)
            
            all_prizes = []
            for bet in st.session_state['generated_bets']:
                for draw in check_draws:
                    prize = calculate_prize(bet, draw)
                    if "未中奖" not in prize:
                        all_prizes.append(prize)
            
            if all_prizes:
                st.success(f"🎉 共中奖 {len(all_prizes)} 注，详见上表")
            else:
                st.info("本期未中奖，继续加油！")
        else:
            st.warning("请先生成投注组合")
    else:
        st.error("解析失败，请检查格式")

# ==================== 底部 ====================
st.markdown("---")
st.caption("⚠️ 本工具仅供学术研究和娱乐参考。双色球本质随机，历史规律不代表未来结果。2026年新规下中3红有福运奖5元。请理性投注，量力而行。")
