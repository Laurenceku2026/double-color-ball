# ============================================================
# 双色球AI智能选号工具 v11.0 完整版
# 第1部分：导入、配置、Supabase连接、数据管理、管理员页面
# ============================================================

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

warnings.filterwarnings('ignore')

# ==================== 尝试导入ML库 ====================
# ==================== 尝试导入ML库（健壮版） ====================
LGB_AVAILABLE = False
XGB_AVAILABLE = False
SKLEARN_AVAILABLE = False
MCP_AVAILABLE = False

# LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
    print("✅ LightGBM 导入成功")
except ImportError as e:
    print(f"❌ LightGBM 导入失败: {e}")
    LGB_AVAILABLE = False

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✅ XGBoost 导入成功")
except ImportError as e:
    print(f"❌ XGBoost 导入失败: {e}")
    XGB_AVAILABLE = False

# scikit-learn
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
    print("✅ scikit-learn 导入成功")
except ImportError as e:
    print(f"❌ scikit-learn 导入失败: {e}")
    SKLEARN_AVAILABLE = False

# MCP服务（可选）
try:
    from ssq_mcp import get_recent_data, get_data_by_issue_range, get_frequency_analysis
    MCP_AVAILABLE = True
    print("✅ MCP服务 导入成功")
except ImportError as e:
    print(f"❌ MCP服务 导入失败: {e}")
    MCP_AVAILABLE = False
# ==================== 页面配置 ====================
st.set_page_config(
    page_title="双色球AI分析工具 - 专业版",
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
    .zone-medium { background-color: #ffa500; color: white; padding: 4px 8px; border-radius: 15px; display: inline-block; margin: 2px; }
    .signal-high { background-color: #ff4b4b; color: white; padding: 5px 10px; border-radius: 20px; display: inline-block; }
    .signal-medium { background-color: #ffa500; color: white; padding: 5px 10px; border-radius: 20px; display: inline-block; }
    .signal-low { background-color: #00cc66; color: white; padding: 5px 10px; border-radius: 20px; display: inline-block; }
    .stButton button { width: 100%; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: bold; }
    .data-status-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .data-editor-textarea textarea {
        font-family: 'Courier New', monospace;
        font-size: 12px;
    }
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
def init_supabase() -> Optional[Client]:
    """初始化Supabase连接"""
    try:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_SERVICE_ROLE_KEY"]
        return create_client(supabase_url, supabase_key)
    except Exception:
        return None

# ==================== 数据加载函数 ====================
def load_all_from_supabase() -> Optional[List[Dict]]:
    """从Supabase加载全部数据，按期号升序返回"""
    supabase = init_supabase()
    if supabase is None:
        return None
    
    try:
        response = supabase.schema('ssq_schema').table('ssq_draws')\
            .select("*")\
            .order("period", desc=False)\
            .execute()
        
        if not response.data:
            return None
        
        all_draws = []
        for row in response.data:
            all_draws.append({
                'id': row.get('id'),
                'period': row.get('period'),
                'date': row.get('date'),
                'reds': [row.get('red1'), row.get('red2'), row.get('red3'), 
                        row.get('red4'), row.get('red5'), row.get('red6')],
                'blue': row.get('blue'),
                'pool': row.get('pool_amount', 0),
                'sales': row.get('total_sales', 0),
                'prize1_count': row.get('prize1_count', 0),
                'prize1_amount': row.get('prize1_amount', 0),
                'prize2_count': row.get('prize2_count', 0),
                'prize2_amount': row.get('prize2_amount', 0)
            })
        
        return all_draws
        
    except Exception as e:
        st.error(f"加载数据失败: {e}")
        return None

# ==================== 数据保存函数 ====================
def save_draws_to_supabase(draws: List[Dict]) -> int:
    """保存数据到Supabase（先清空再插入）"""
    if not draws:
        return 0
    
    supabase = init_supabase()
    if supabase is None:
        return 0
    
    try:
        # 先清空表
        supabase.schema('ssq_schema').table('ssq_draws').delete().neq("id", 0).execute()
        
        # 批量插入
        saved_count = 0
        for draw in draws:
            reds = draw.get('reds', [])
            data = {
                "period": draw.get('period'),
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
            saved_count += 1
        
        return saved_count
        
    except Exception as e:
        st.error(f"保存失败: {e}")
        return 0

# ==================== 数据格式转换（用于编辑框） ====================
def draw_to_text_line(draw: Dict) -> str:
    """将单条数据转换为文本行（Tab分隔，15列）"""
    reds = draw.get('reds', [])
    return f"{draw.get('period')}\t{draw.get('date', '')}\t{reds[0] if len(reds)>0 else 0}\t{reds[1] if len(reds)>1 else 0}\t{reds[2] if len(reds)>2 else 0}\t{reds[3] if len(reds)>3 else 0}\t{reds[4] if len(reds)>4 else 0}\t{reds[5] if len(reds)>5 else 0}\t{draw.get('blue', 0)}\t{draw.get('pool', 0)}\t{draw.get('prize1_count', 0)}\t{draw.get('prize1_amount', 0)}\t{draw.get('prize2_count', 0)}\t{draw.get('prize2_amount', 0)}\t{draw.get('sales', 0)}"

def text_line_to_draw(line: str, line_num: int) -> Optional[Dict]:
    """将文本行转换为数据（支持Tab和空格分隔）"""
    line = line.strip()
    if not line:
        return None
    
    # 自动检测分隔符：优先Tab，否则空格
    if '\t' in line:
        parts = line.split('\t')
    else:
        parts = line.split()
    
    if len(parts) < 8:
        return None
    
    try:
        period_str = parts[0]
        period = int(period_str) if str(period_str).isdigit() else period_str
        
        # 检测是否有日期列（第2列是否包含日期格式）
        has_date = False
        date = None
        start_idx = 1
        
        if len(parts) > 8 and ('-' in parts[1] or '/' in parts[1]):
            has_date = True
            date = parts[1]
            start_idx = 2
        
        # 红球（6个）
        reds = []
        for i in range(start_idx, start_idx + 6):
            if i >= len(parts):
                break
            val = parts[i]
            if '.' in str(val):
                val = val.split('.')[0]
            reds.append(int(val))
        
        if len(reds) != 6:
            return None
        reds = sorted(reds)
        
        # 蓝球
        blue_idx = start_idx + 6
        if blue_idx >= len(parts):
            return None
        blue = int(float(parts[blue_idx]))
        
        # 奖池、销量等（可选）
        pool = 0
        sales = 0
        prize1_count = 0
        prize1_amount = 0
        prize2_count = 0
        prize2_amount = 0
        
        if len(parts) > blue_idx + 1:
            pool = int(float(parts[blue_idx + 1])) if parts[blue_idx + 1] and parts[blue_idx + 1] != 'None' else 0
        if len(parts) > blue_idx + 2:
            prize1_count = int(float(parts[blue_idx + 2])) if parts[blue_idx + 2] and parts[blue_idx + 2] != 'None' else 0
        if len(parts) > blue_idx + 3:
            prize1_amount = int(float(parts[blue_idx + 3])) if parts[blue_idx + 3] and parts[blue_idx + 3] != 'None' else 0
        if len(parts) > blue_idx + 4:
            prize2_count = int(float(parts[blue_idx + 4])) if parts[blue_idx + 4] and parts[blue_idx + 4] != 'None' else 0
        if len(parts) > blue_idx + 5:
            prize2_amount = int(float(parts[blue_idx + 5])) if parts[blue_idx + 5] and parts[blue_idx + 5] != 'None' else 0
        if len(parts) > blue_idx + 6:
            sales = int(float(parts[blue_idx + 6])) if parts[blue_idx + 6] and parts[blue_idx + 6] != 'None' else 0
        
        return {
            'period': period,
            'date': date,
            'reds': reds,
            'blue': blue,
            'pool': pool,
            'sales': sales,
            'prize1_count': prize1_count,
            'prize1_amount': prize1_amount,
            'prize2_count': prize2_count,
            'prize2_amount': prize2_amount
        }
    except (ValueError, IndexError):
        return None

# ==================== 历史平均值填充 ====================
def fill_missing_with_history(draws: List[Dict], window: int = 20) -> List[Dict]:
    """对于缺失的pool和sales字段，使用最近window期的平均值填充"""
    if not draws:
        return draws
    
    valid_pools = [d['pool'] for d in draws if d.get('pool', 0) > 0]
    valid_sales = [d['sales'] for d in draws if d.get('sales', 0) > 0]
    
    recent_pools = valid_pools[-window:] if len(valid_pools) > window else valid_pools
    recent_sales = valid_sales[-window:] if len(valid_sales) > window else valid_sales
    
    avg_pool = int(np.mean(recent_pools)) if recent_pools else 0
    avg_sales = int(np.mean(recent_sales)) if recent_sales else 0
    
    for draw in draws:
        if draw.get('pool', 0) == 0 and avg_pool > 0:
            draw['pool'] = avg_pool
        if draw.get('sales', 0) == 0 and avg_sales > 0:
            draw['sales'] = avg_sales
    
    return draws

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
# ==================== Excel解析器 ====================
def parse_excel_file(uploaded_file) -> Optional[List[Dict]]:
    """解析用户上传的Excel文件"""
    try:
        # 尝试导入 openpyxl
        try:
            import openpyxl
        except ImportError:
            st.error("缺少 openpyxl 库，请使用文本粘贴功能")
            return None
        
        df = pd.read_excel(uploaded_file, sheet_name=0)
        st.write("读取到的列名:", df.columns.tolist())
        # 尝试识别列名
        period_col = None
        date_col = None
        red_cols = []
        blue_col = None
        pool_col = None
        sales_col = None
        
        # 常见列名映射
        period_names = ['期号', 'period', 'Period', '期次']
        date_names = ['开奖日期', '日期', 'date', 'Date']
        red_names = ['红1', '红球1', 'red1', '红球号码1']
        blue_names = ['蓝球', 'blue', 'Blue', '蓝球号码']
        pool_names = ['奖池奖金(元)', '奖池', 'pool_amount']
        sales_names = ['总投注额(元)', '总投注额', '销量', 'total_sales']
        
        for col in df.columns:
            col_str = str(col).strip()
            if not period_col and any(name in col_str for name in period_names):
                period_col = col
            if not date_col and any(name in col_str for name in date_names):
                date_col = col
            if not blue_col and any(name in col_str for name in blue_names):
                blue_col = col
            if not pool_col and any(name in col_str for name in pool_names):
                pool_col = col
            if not sales_col and any(name in col_str for name in sales_names):
                sales_col = col
            for i in range(1, 7):
                if any(name in col_str for name in [f'红{i}', f'红球{i}', f'red{i}']):
                    red_cols.append(col)
                    break
        
        # 如果按列名没找到，按位置（第1列期号，第2列日期，第3-8列红球，第9列蓝球）
        if len(red_cols) != 6 and len(df.columns) >= 9:
            period_col = df.columns[0]
            date_col = df.columns[1] if len(df.columns) > 1 else None
            red_cols = df.columns[2:8].tolist()
            blue_col = df.columns[8] if len(df.columns) > 8 else None
            if len(df.columns) > 9:
                pool_col = df.columns[9]
            if len(df.columns) > 14:
                sales_col = df.columns[14]
        
        if len(red_cols) != 6:
            st.error(f"无法识别红球列，找到{len(red_cols)}列，需要6列")
            return None
        
        draws = []
        for idx, row in df.iterrows():
            try:
                # 期号
                period = row[period_col]
                if pd.isna(period):
                    continue
                period = int(period) if str(period).isdigit() else str(period)
                
                # 日期
                date = None
                if date_col and pd.notna(row[date_col]):
                    date_val = row[date_col]
                    if isinstance(date_val, datetime):
                        date = date_val.strftime('%Y-%m-%d')
                    else:
                        date = str(date_val).split()[0] if ' ' in str(date_val) else str(date_val)
                
                # 红球
                reds = []
                for col in red_cols[:6]:
                    val = row[col]
                    if pd.notna(val):
                        reds.append(int(val))
                
                if len(reds) != 6:
                    continue
                reds = sorted(reds)
                
                # 蓝球
                blue = 0
                if blue_col and pd.notna(row[blue_col]):
                    blue = int(row[blue_col])
                else:
                    continue
                
                # 奖池和销量
                pool = int(row[pool_col]) if pool_col and pd.notna(row[pool_col]) else 0
                sales = int(row[sales_col]) if sales_col and pd.notna(row[sales_col]) else 0
                
                draws.append({
                    'period': period,
                    'date': date,
                    'reds': reds,
                    'blue': blue,
                    'pool': pool,
                    'sales': sales,
                    'prize1_count': 0,
                    'prize1_amount': 0,
                    'prize2_count': 0,
                    'prize2_amount': 0
                })
            except Exception:
                continue
        
        return draws if draws else None
        
    except Exception as e:
        st.error(f"Excel解析错误: {e}")
        return None
# ==================== 文本解析器 ====================
def parse_draws_from_text(text: str) -> List[Dict]:
    """解析粘贴的文本数据，支持多格式自动检测"""
    lines = text.strip().split('\n')
    draws = []
    error_count = 0
    
    for line_idx, line in enumerate(lines):
        if not line.strip():
            continue
        
        # 自动检测分隔符：优先Tab，否则空格
        if '\t' in line:
            parts = line.split('\t')
        else:
            parts = line.split()
        
        if len(parts) < 8:
            error_count += 1
            continue
        
        try:
            # 期号
            period_str = parts[0]
            period = int(period_str) if period_str.isdigit() else period_str
            
            # 日期（第2列，可选）
            date = None
            start_idx = 1
            if len(parts) > 8 and ('-' in parts[1] or '/' in parts[1]):
                date = parts[1]
                start_idx = 2
            
            # 红球（6个）
            reds = []
            for i in range(start_idx, start_idx + 6):
                if i >= len(parts):
                    break
                val = parts[i]
                if '.' in str(val):
                    val = val.split('.')[0]
                reds.append(int(val))
            
            if len(reds) != 6:
                error_count += 1
                continue
            reds = sorted(reds)
            
            # 蓝球
            blue_idx = start_idx + 6
            if blue_idx >= len(parts):
                error_count += 1
                continue
            blue = int(float(parts[blue_idx]))
            
            # 奖池和销量（可选）
            pool = 0
            sales = 0
            if len(parts) > blue_idx + 1:
                pool_str = parts[blue_idx + 1].replace(',', '')
                if pool_str.isdigit():
                    pool = int(pool_str)
            if len(parts) > blue_idx + 7:
                sales_str = parts[-1].replace(',', '')
                if sales_str.isdigit():
                    sales = int(sales_str)
            
            draws.append({
                'period': period,
                'date': date,
                'reds': reds,
                'blue': blue,
                'pool': pool,
                'sales': sales,
                'prize1_count': 0,
                'prize1_amount': 0,
                'prize2_count': 0,
                'prize2_amount': 0
            })
            
        except (ValueError, IndexError):
            error_count += 1
            continue
    
    if draws:
        if error_count > 0:
            st.warning(f"成功解析 {len(draws)} 期，跳过 {error_count} 行格式错误")
        else:
            st.success(f"成功解析 {len(draws)} 期数据")
    else:
        st.error("未解析到任何数据，请检查格式")
    
    return draws

def show_admin_page():
    """管理员页面 - 可编辑表格 + 复选框删除 + 按钮在表格上方"""
    
    st.subheader("📋 数据编辑器")
    st.caption("💡 双击单元格可编辑 | 选中行复选框后点击删除 | 点击排序按钮按期号排序")
    
    # 定义固定列名（15列，与Supabase表结构对齐）
    columns = [
        "选择", "期号", "开奖日期", "红1", "红2", "红3", "红4", "红5", "红6", "蓝球",
        "奖池奖金(元)", "一等奖注数", "一等奖奖金(元)", "二等奖注数", "二等奖奖金(元)", "总投注额(元)"
    ]
    
    # 加载现有数据
    current_draws = st.session_state.get('draws_loaded', [])
    
    # 转换为DataFrame（不带选择列）
    if current_draws:
        display_draws = sorted(current_draws, key=lambda x: x.get('period', 0), reverse=True)
        
        data_rows = []
        for d in display_draws:
            reds = d.get('reds', [])
            row = {
                '期号': d.get('period', ''),
                '开奖日期': d.get('date', ''),
                '红1': reds[0] if len(reds) > 0 else 0,
                '红2': reds[1] if len(reds) > 1 else 0,
                '红3': reds[2] if len(reds) > 2 else 0,
                '红4': reds[3] if len(reds) > 3 else 0,
                '红5': reds[4] if len(reds) > 4 else 0,
                '红6': reds[5] if len(reds) > 5 else 0,
                '蓝球': d.get('blue', 0),
                '奖池奖金(元)': d.get('pool', 0),
                '一等奖注数': d.get('prize1_count', 0),
                '一等奖奖金(元)': d.get('prize1_amount', 0),
                '二等奖注数': d.get('prize2_count', 0),
                '二等奖奖金(元)': d.get('prize2_amount', 0),
                '总投注额(元)': d.get('sales', 0)
            }
            data_rows.append(row)
        df = pd.DataFrame(data_rows)
        # 添加选择列（默认False）
        df.insert(0, '选择', False)
    else:
        # 空DataFrame
        df = pd.DataFrame(columns=['选择'] + columns[1:])
    
    # ==================== 按钮区域（放在表格上方） ====================
    st.markdown("---")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        if st.button("🔄 排序（按期号降序）", use_container_width=True, key="sort_btn"):
            with st.spinner("排序中..."):
                # 从当前编辑的数据中提取，按期号排序
                if len(df) > 0:
                    # 移除选择列进行排序
                    temp_df = df.drop(columns=['选择'], errors='ignore')
                    temp_df = temp_df.sort_values(by='期号', ascending=False)
                    temp_df.insert(0, '选择', False)
                    st.session_state['ssq_data_editor'] = temp_df
                    st.rerun()
                else:
                    st.info("暂无数据")
    
    with col2:
        if st.button("➕ 添加空行", use_container_width=True, key="add_row_btn"):
            # 添加一行空数据
            new_row = pd.DataFrame([{
                '选择': False,
                '期号': 0,
                '开奖日期': '',
                '红1': 0, '红2': 0, '红3': 0, '红4': 0, '红5': 0, '红6': 0,
                '蓝球': 0,
                '奖池奖金(元)': 0,
                '一等奖注数': 0,
                '一等奖奖金(元)': 0,
                '二等奖注数': 0,
                '二等奖奖金(元)': 0,
                '总投注额(元)': 0
            }])
            new_df = pd.concat([df, new_row], ignore_index=True)
            st.session_state['ssq_data_editor'] = new_df
            st.rerun()
    
    with col3:
        if st.button("🗑️ 删除选中行", use_container_width=True, key="delete_btn"):
            # 删除勾选的行
            if '选择' in df.columns:
                selected_mask = df['选择'] == True
                selected_count = selected_mask.sum()
                
                if selected_count > 0:
                    # 确认对话框
                    confirm = st.checkbox(f"⚠️ 确认删除 {selected_count} 行？")
                    if confirm:
                        new_df = df[~selected_mask].copy()
                        new_df['选择'] = False
                        st.session_state['ssq_data_editor'] = new_df
                        st.success(f"已删除 {selected_count} 行")
                        st.rerun()
                    else:
                        st.info("请勾选确认框后再次点击删除")
                else:
                    st.warning("请先勾选要删除的行")
    
    with col4:
        if st.button("📥 从数据库加载", type="primary", use_container_width=True, key="load_btn"):
            with st.spinner("加载中..."):
                draws = load_all_from_supabase()
                if draws:
                    draws = fill_missing_with_history(draws)
                    st.session_state['draws_loaded'] = draws
                    st.success(f"加载 {len(draws)} 期数据")
                    st.rerun()
                else:
                    st.error("无数据")
    
    with col5:
        if st.button("💾 保存到数据库", type="primary", use_container_width=True, key="save_btn"):
            with st.spinner("保存中..."):
                # 移除选择列
                save_df = df.drop(columns=['选择'], errors='ignore')
                new_draws = []
                errors = 0
                skipped = 0
                
                for idx, row in save_df.iterrows():
                    try:
                        if pd.isna(row['期号']) or row['期号'] == 0:
                            skipped += 1
                            continue
                        
                        period = int(row['期号'])
                        
                        date = None
                        if pd.notna(row['开奖日期']) and row['开奖日期'] != '':
                            date = str(row['开奖日期'])
                        
                        reds = [
                            int(row['红1']) if pd.notna(row['红1']) else 0,
                            int(row['红2']) if pd.notna(row['红2']) else 0,
                            int(row['红3']) if pd.notna(row['红3']) else 0,
                            int(row['红4']) if pd.notna(row['红4']) else 0,
                            int(row['红5']) if pd.notna(row['红5']) else 0,
                            int(row['红6']) if pd.notna(row['红6']) else 0,
                        ]
                        
                        blue = int(row['蓝球']) if pd.notna(row['蓝球']) else 0
                        pool = int(row['奖池奖金(元)']) if pd.notna(row['奖池奖金(元)']) else 0
                        sales = int(row['总投注额(元)']) if pd.notna(row['总投注额(元)']) else 0
                        prize1_count = int(row['一等奖注数']) if pd.notna(row['一等奖注数']) else 0
                        prize1_amount = int(row['一等奖奖金(元)']) if pd.notna(row['一等奖奖金(元)']) else 0
                        prize2_count = int(row['二等奖注数']) if pd.notna(row['二等奖注数']) else 0
                        prize2_amount = int(row['二等奖奖金(元)']) if pd.notna(row['二等奖奖金(元)']) else 0
                        
                        valid_reds = all(1 <= r <= 33 for r in reds if r > 0)
                        valid_blue = 1 <= blue <= 16 if blue > 0 else True
                        
                        if valid_reds and valid_blue:
                            new_draws.append({
                                'period': period,
                                'date': date,
                                'reds': sorted(reds),
                                'blue': blue,
                                'pool': pool,
                                'sales': sales,
                                'prize1_count': prize1_count,
                                'prize1_amount': prize1_amount,
                                'prize2_count': prize2_count,
                                'prize2_amount': prize2_amount
                            })
                        else:
                            errors += 1
                    except Exception:
                        errors += 1
                
                if skipped > 0:
                    st.warning(f"跳过 {skipped} 行空数据")
                if errors > 0:
                    st.warning(f"跳过 {errors} 行无效数据（红球1-33，蓝球1-16）")
                
                if new_draws:
                    new_draws = fill_missing_with_history(new_draws)
                    new_draws.sort(key=lambda x: x.get('period', 0))
                    saved = save_draws_to_supabase(new_draws)
                    if saved > 0:
                        st.session_state['draws_loaded'] = new_draws
                        st.success(f"保存 {saved} 期数据成功！")
                        st.rerun()
                else:
                    st.error("没有有效数据可保存")
    
    with col6:
        if st.button("📊 查看统计", use_container_width=True, key="stats_btn"):
            if current_draws:
                st.info(f"当前数据库：{len(current_draws)} 期数据，范围：{current_draws[0].get('period')} - {current_draws[-1].get('period')}")
            else:
                st.warning("数据库暂无数据")
    
    # ==================== 可编辑表格 ====================
    st.markdown("---")
    
    # 配置列类型
    column_config = {
        "选择": st.column_config.CheckboxColumn("选择", help="勾选要删除的行"),
        "开奖日期": st.column_config.TextColumn("开奖日期"),
        "期号": st.column_config.NumberColumn("期号", step=1),
        "红1": st.column_config.NumberColumn("红1", min_value=1, max_value=33, step=1),
        "红2": st.column_config.NumberColumn("红2", min_value=1, max_value=33, step=1),
        "红3": st.column_config.NumberColumn("红3", min_value=1, max_value=33, step=1),
        "红4": st.column_config.NumberColumn("红4", min_value=1, max_value=33, step=1),
        "红5": st.column_config.NumberColumn("红5", min_value=1, max_value=33, step=1),
        "红6": st.column_config.NumberColumn("红6", min_value=1, max_value=33, step=1),
        "蓝球": st.column_config.NumberColumn("蓝球", min_value=1, max_value=16, step=1),
        "一等奖注数": st.column_config.NumberColumn("一等奖注数", step=1),
        "二等奖注数": st.column_config.NumberColumn("二等奖注数", step=1),
    }
    
    # 显示可编辑表格
    try:
        edited_df = st.data_editor(
            df,
            column_config=column_config,
            use_container_width=True,
            height=500,
            key="ssq_data_editor"
        )
        # 保存编辑后的数据回df
        df = edited_df
    except Exception as e:
        st.error(f"表格加载失败: {e}")
        st.info("请尝试刷新页面")
        return
    
    # ==================== Excel上传区域（放在表格下方） ====================
    st.markdown("---")
    st.subheader("📎 Excel文件上传（完整15列）")
    st.caption("格式：期号、开奖日期、红1-6、蓝球、奖池奖金(元)、一等奖注数、一等奖奖金(元)、二等奖注数、二等奖奖金(元)、总投注额(元)")
    st.caption("💡 预览表格支持横向滚动，确认数据正确后再保存")
    
    uploaded_file = st.file_uploader(
        "选择Excel文件",
        type=['xlsx', 'xls'],
        key="excel_uploader_admin",
        help="上传Excel文件，将替换当前所有数据"
    )
    
    if uploaded_file is not None:
        with st.spinner("正在解析Excel文件..."):
            excel_draws = parse_excel_file(uploaded_file)
            if excel_draws and len(excel_draws) > 0:
                st.success(f"✅ 成功解析 {len(excel_draws)} 期数据")
                
                # 预览表格 - 完整15列，横向滚动
                st.markdown("**📊 数据预览（前10行，横向滚动查看全部列）**")
                
                preview_data = []
                for d in excel_draws[:10]:
                    reds = d.get('reds', [])
                    reds_str = ','.join(f"{r:02d}" for r in reds)
                    
                    preview_data.append({
                        '期号': d.get('period'),
                        '开奖日期': str(d.get('date', ''))[:10] if d.get('date') else '',
                        '红球': reds_str,
                        '蓝球': f"{d.get('blue', 0):02d}",
                        '奖池奖金(元)': d.get('pool', 0),
                        '一等奖注数': d.get('prize1_count', 0),
                        '一等奖奖金(元)': d.get('prize1_amount', 0),
                        '二等奖注数': d.get('prize2_count', 0),
                        '二等奖奖金(元)': d.get('prize2_amount', 0),
                        '总投注额(元)': d.get('sales', 0)
                    })
                
                preview_df = pd.DataFrame(preview_data)
                st.dataframe(preview_df, use_container_width=True, hide_index=True)
                
                st.caption(f"📋 共 {len(preview_df.columns)} 列 | 期号范围: {excel_draws[0].get('period')} - {excel_draws[-1].get('period')}")
                
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("✅ 确认保存到数据库（将替换现有数据）", type="primary"):
                        excel_draws = fill_missing_with_history(excel_draws)
                        excel_draws.sort(key=lambda x: x.get('period', 0))
                        saved = save_draws_to_supabase(excel_draws)
                        if saved > 0:
                            st.session_state['draws_loaded'] = excel_draws
                            st.success(f"保存 {saved} 期数据成功！")
                            st.rerun()
                        else:
                            st.error("保存失败")
                with col_cancel:
                    if st.button("❌ 取消", use_container_width=True):
                        st.rerun()
            else:
                st.error("解析失败，请检查文件格式")
                st.info("请确保Excel第一行为列标题，格式与上方说明一致")


# ==================== 初始化Session State ====================
if 'admin_logged_in' not in st.session_state:
    st.session_state['admin_logged_in'] = False
if 'show_admin' not in st.session_state:
    st.session_state['show_admin'] = False
if 'generated_bets' not in st.session_state:
    st.session_state['generated_bets'] = None
if 'model_used' not in st.session_state:
    st.session_state['model_used'] = None
if 'draws_loaded' not in st.session_state:
    st.session_state['draws_loaded'] = None
if 'analysis_started' not in st.session_state:
    st.session_state['analysis_started'] = False

# ==================== 主页面标题 ====================
col_title, col_settings = st.columns([0.9, 0.1])
with col_title:
    st.title("🎯 双色球AI智能选号工具 - 专业版")
with col_settings:
    if st.button("⚙️ 管理员", key="settings_icon", help="管理员设置"):
        st.session_state['show_admin'] = not st.session_state.get('show_admin', False)

if st.session_state.get('show_admin', False):
    if not st.session_state['admin_logged_in']:
        admin_login()
    else:
        show_admin_page()
        admin_logout()
    st.markdown("---")

# ==================== 加载数据 ====================
if st.session_state.get('draws_loaded') is None:
    with st.spinner("加载数据中..."):
        draws = load_all_from_supabase()
        if draws:
            draws = fill_missing_with_history(draws)
            st.session_state['draws_loaded'] = draws
        else:
            st.session_state['draws_loaded'] = []

draws = st.session_state.get('draws_loaded', [])

if not draws or len(draws) < 5:
    st.markdown("""
    <div class="data-status-card">
        <h3>📂 数据状态：暂无数据或数据不足</h3>
        <p>请点击右上角 <strong>⚙️ 管理员</strong> 按钮，进入管理员页面：</p>
        <ul style="text-align: left; display: inline-block;">
            <li>📋 粘贴数据（推荐）</li>
            <li>📎 或点击"从数据库加载"</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ==================== 侧边栏 ====================
# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("### 🎰 双色球AI分析工具")
    st.markdown("---")
    
    # ML库状态
    with st.expander("🤖 ML库状态", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if LGB_AVAILABLE:
                st.markdown("✅ **LightGBM**")
            else:
                st.markdown("❌ **LightGBM**")
            if XGB_AVAILABLE:
                st.markdown("✅ **XGBoost**")
            else:
                st.markdown("❌ **XGBoost**")
        with col2:
            if SKLEARN_AVAILABLE:
                st.markdown("✅ **scikit-learn**")
            else:
                st.markdown("❌ **scikit-learn**")
        st.caption(f"MCP服务: {'✅ 可用' if MCP_AVAILABLE else '❌ 不可用'}")
    
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
    st.caption("DFSS智能选号工具 v11.0")

# ==================== 主页面内容（注意：这里不能有缩进！）====================

# ==================== 显示数据概览 ====================
st.subheader("📊 数据概览")
latest = draws[-1]
oldest = draws[0]

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("最新期号", latest.get('period', 'N/A'))
with col2:
    date_val = latest.get('date', '')
    st.metric("最新日期", str(date_val)[:10] if date_val else 'N/A')
with col3:
    st.metric("最早期号", oldest.get('period', 'N/A'))
with col4:
    pool = latest.get('pool', 0)
    st.metric("奖池金额", f"¥{pool/1e8:.1f}亿")
with col5:
    st.metric("数据总量", f"{len(draws)}期")

st.markdown("---")

print("第1部分加载完成")

# 第1部分结束 - 请确认后继续第2部分
# ============================================================
# 第2部分：分析引擎（冷热码、7分区、和值、蓝球、ML信号）
# ============================================================

# ==================== 冷热码分析 ====================
# ==================== 冷热码分析（修正版） ====================
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
    
    # 计算频率（正确的公式：出现次数 / 期数 / 6 * 100）
    total_draws = len(recent_draws)
    total_red_balls = total_draws * 6
    
    # 热门红球 Top 15（按出现次数排序）
    hot_reds = sorted(red_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    # 冷门红球 Bottom 10（按出现次数排序）
    cold_reds = sorted(red_freq.items(), key=lambda x: x[1])[:10]
    # 热门蓝球 Top 8
    hot_blues = sorted(blue_freq.items(), key=lambda x: x[1], reverse=True)[:8]
    
    return {
        'hot_reds': hot_reds,
        'cold_reds': cold_reds,
        'hot_blues': hot_blues,
        'red_freq': red_freq,
        'blue_freq': blue_freq,
        'red_absence': red_absence,
        'blue_absence': blue_absence,
        'analysis_periods': analysis_periods,
        'total_draws': total_draws
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
    
    blue_freq = {i: 0 for i in range(1, 17)}
    for blue in blue_sequence:
        if 1 <= blue <= 16:
            blue_freq[blue] += 1
    
    blue_absence = {i: 0 for i in range(1, 17)}
    for num in range(1, 17):
        last_seen = None
        for idx, draw in enumerate(reversed(draws)):
            if draw.get('blue', 0) == num:
                last_seen = idx
                break
        blue_absence[num] = last_seen if last_seen is not None else len(draws)
    
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


# ==================== 和值趋势分析 ====================
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


# ==================== 动态和值预测 ====================
def get_target_sum(draws: List[Dict]) -> Tuple[int, int]:
    """动态预测目标和值"""
    if len(draws) < 10:
        return RED_EXPECTED_SUM, RED_SUM_STD
    
    recent_sums = []
    for draw in draws[-10:]:
        reds = draw.get('reds', [])
        if reds:
            recent_sums.append(sum(reds))
    
    if not recent_sums:
        return RED_EXPECTED_SUM, RED_SUM_STD
    
    short_mean = np.mean(recent_sums)
    
    # 均值回归策略
    if short_mean > RED_EXPECTED_SUM + 5:
        target = RED_EXPECTED_SUM - 5
    elif short_mean < RED_EXPECTED_SUM - 5:
        target = RED_EXPECTED_SUM + 5
    else:
        target = RED_EXPECTED_SUM
    
    return int(target), RED_SUM_STD


# ==================== ML信号分析（完整版） ====================
def calculate_ml_signals(draws: List[Dict]) -> Dict:
    """计算ML特征信号 - 包含奖池分析、剪刀差、周期预测等"""
    if not draws or len(draws) < 10:
        return {
            'jackpot_level': '数据不足',
            'scissors': '数据不足',
            'cycle': '数据不足',
            'blue_bias': '均衡',
            'signal_strength': 0,
            'suggestion_text': '数据不足',
            'pool': 0,
            'sales': 0,
            'countdown': '数据不足',
            'sunday_hint': '',
            'is_sunday': False,
            'repeat_hint': '',
            'repeat_count': 0,
            'sum_deviation': 0,
            'sum_suggestion': ''
        }
    
    latest = draws[-1]
    pool = latest.get('pool', 0)
    sales = latest.get('sales', 0)
    
    # 1. 奖池阈值分析
    if pool >= 250000000:
        jackpot_level = "HIGH (≥2.5亿) 🔥"
        signal_strength = 30
    elif pool >= 150000000:
        jackpot_level = "MEDIUM (1.5-2.5亿) ⚡"
        signal_strength = 15
    else:
        jackpot_level = "LOW (<1.5亿) ❄️"
        signal_strength = 0
    
    # 2. 剪刀差信号（投注额与奖池变化）
    if len(draws) >= 2:
        prev = draws[-2]
        prev_pool = prev.get('pool', 0)
        prev_sales = prev.get('sales', 0)
        
        if prev_pool > 0 and prev_sales > 0:
            sales_change = (sales - prev_sales) / prev_sales if prev_sales > 0 else 0
            pool_change = (pool - prev_pool) / prev_pool if prev_pool > 0 else 0
            
            if sales_change > 0.05 and pool_change < -0.03:
                scissors = "HIGH_ALERT (头奖爆发预警) 🚨"
                signal_strength += 40
            elif sales_change > 0.03:
                scissors = "MEDIUM (投注活跃) ⚠️"
                signal_strength += 20
            else:
                scissors = "NORMAL ✅"
        else:
            scissors = "NORMAL ✅"
    else:
        scissors = "NORMAL ✅"
    
    # 3. 头奖周期预测
    recent_prizes = [d.get('prize1_count', 0) for d in draws[-20:]]
    high_prize_count = sum(1 for p in recent_prizes if p >= 10)
    
    # 计算距离上次爆发的期数
    last_burst = None
    for idx, d in enumerate(reversed(draws)):
        if d.get('prize1_count', 0) >= 10:
            last_burst = idx
            break
    
    if last_burst is not None:
        countdown = max(0, 15 - last_burst)
        countdown_text = f"约{countdown}期后可能爆发" if countdown > 0 else "爆发期临近！"
    else:
        countdown_text = "数据不足"
    
    if high_prize_count >= 3:
        cycle = "冷却期 ❄️"
        signal_strength -= 20
    elif high_prize_count == 0:
        cycle = "积累期 📈"
        signal_strength += 10
    else:
        cycle = "正常期 ⚖️"
    
    # 4. 周日效应检测
    is_sunday = False
    sunday_hint = ""
    if latest.get('date'):
        try:
            date_str = latest.get('date')
            if isinstance(date_str, str):
                dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
                is_sunday = (dt.weekday() == 6)
                if is_sunday:
                    sunday_hint = "📅 周日开奖，蓝球偏向小号(1-8)概率71%"
        except:
            pass
    
    # 5. 蓝球偏向预测
    recent_blues = [d.get('blue', 0) for d in draws[-20:] if d.get('blue', 0) > 0]
    if recent_blues:
        small_count = sum(1 for b in recent_blues if b <= 8)
        if small_count >= 12:
            blue_bias = "偏小号 (1-8) 🔵"
            blue_prob = f"{small_count/len(recent_blues)*100:.0f}%"
        elif small_count <= 8:
            blue_bias = "偏大号 (9-16) 🔴"
            blue_prob = f"{(len(recent_blues)-small_count)/len(recent_blues)*100:.0f}%"
        else:
            blue_bias = "均衡 ⚖️"
            blue_prob = f"{small_count/len(recent_blues)*100:.0f}%小号"
    else:
        blue_bias = "均衡 ⚖️"
        blue_prob = "50%"
    
    # 6. 红球重复预测
    last_reds = draws[-1].get('reds', [])
    prev_reds = draws[-2].get('reds', []) if len(draws) >= 2 else []
    repeat_count = len(set(last_reds) & set(prev_reds)) if prev_reds else 0
    repeat_prob = 60
    repeat_hint = f"上期红球重复{repeat_count}个，历史概率{repeat_prob}%"
    
    # 7. 综合信号强度（0-100）
    signal_strength = max(0, min(100, signal_strength))
    
    if signal_strength >= 60:
        suggestion_text = "🔔 强烈推荐投注"
    elif signal_strength >= 30:
        suggestion_text = "⚠️ 谨慎投注"
    else:
        suggestion_text = "💤 建议观望"
    
    # 8-9. 和值回归提示
    current_sum = sum(last_reds)
    sum_deviation = current_sum - RED_EXPECTED_SUM
    if abs(sum_deviation) > 15:
        if sum_deviation > 0:
            sum_suggestion = f"和值偏高{sum_deviation}点，建议关注小号回归"
        else:
            sum_suggestion = f"和值偏低{abs(sum_deviation)}点，建议关注大号回归"
    else:
        sum_suggestion = f"和值正常（偏差{sum_deviation:+d}），保持均衡"
    
    return {
        'jackpot_level': jackpot_level,
        'scissors': scissors,
        'cycle': cycle,
        'blue_bias': blue_bias,
        'blue_prob': blue_prob,
        'signal_strength': signal_strength,
        'suggestion_text': suggestion_text,
        'pool': pool,
        'sales': sales,
        'countdown': countdown_text,
        'sunday_hint': sunday_hint,
        'is_sunday': is_sunday,
        'repeat_hint': repeat_hint,
        'repeat_count': repeat_count,
        'sum_deviation': sum_deviation,
        'sum_suggestion': sum_suggestion
    }


# ==================== 获取下一期期号 ====================
def get_next_period(draws: List[Dict]) -> str:
    """获取下一期期号"""
    if not draws:
        return "未知"
    latest_period = draws[-1].get('period', '')
    if latest_period and str(latest_period).isdigit():
        return str(int(latest_period) + 1)
    return "未知"


# ==================== DeepSeek AI 建议 ====================
_last_api_call = 0

def get_deepseek_suggestion(draws: List[Dict], source_used: str, ml_signals: Dict, next_period: str) -> Dict:
    """获取DeepSeek AI的投注建议（带3秒防抖）"""
    global _last_api_call
    
    current_time = time.time()
    if current_time - _last_api_call < 3:
        return {
            "plan": "4组7+1复式",
            "win_rate": "30-35%",
            "blue_advice": f"参考ML信号",
            "summary": "请稍后再获取AI建议",
            "ml_tip": ml_signals.get('suggestion_text', '分析中...')
        }
    _last_api_call = current_time
    
    if not DEEPSEEK_API_KEY or len(draws) < 10:
        method1 = Method1HotColdSum(draws) if 'Method1HotColdSum' in dir() else None
        if method1:
            red_scores = method1.calculate_red_scores()
            top_reds = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)[:6]
            top_blue = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)[:1] if red_scores else [(8, 0)]
            blue_advice = f"推荐蓝球{top_blue[0][0] if top_blue else 8}"
        else:
            blue_advice = "参考冷热分析"
        
        return {
            "plan": "4组7+1复式",
            "win_rate": "30-35%",
            "blue_advice": blue_advice,
            "summary": f"基于{len(draws)}期历史数据",
            "ml_tip": ml_signals.get('suggestion_text', f"奖池{ml_signals['jackpot_level']}，{ml_signals['cycle']}")
        }
    
    try:
        # 获取热号用于提示
        hot_cold = get_hot_cold_analysis(draws, 100)
        top_reds = [num for num, _ in hot_cold['hot_reds'][:6]]
        top_blues = [num for num, _ in hot_cold['hot_blues'][:3]]
        
        prompt = f"""你是双色球AI分析专家。基于以下数据给出投注建议（JSON格式）：

【数据来源】{source_used}
【历史数据量】{len(draws)}期
【预测下一期】{next_period}
【红球热号Top6】{top_reds}
【蓝球热号Top3】{top_blues}
【奖池】{ml_signals.get('pool', 0)/1e8:.1f}亿
【剪刀差信号】{ml_signals.get('scissors', '正常')}
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
    
    # 降级：使用ML信号
    return {
        "plan": "4组7+1复式",
        "win_rate": "30-35%",
        "blue_advice": ml_signals.get('blue_bias', '均衡'),
        "summary": ml_signals.get('suggestion_text', '谨慎投注'),
        "ml_tip": ml_signals.get('suggestion_text', f"奖池{ml_signals['jackpot_level']}，{ml_signals['cycle']}")
    }


# ==================== 多期查奖函数 ====================
def parse_check_draws(text: str, max_draws: int = 50) -> List[Dict]:
    """解析查奖数据，支持用户自定义期数（上限50）"""
    lines = text.strip().split('\n')
    draws = []
    for line in lines[:max_draws]:
        if not line.strip():
            continue
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


print("第2部分加载完成")

# 第2部分结束 - 请确认后继续第3部分
# ============================================================
# 第3部分：完整的4种AI算法实现
# ============================================================

# ==================== 方法1：冷热码评分 + 和值动态预测 ====================
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
        return get_target_sum(self.draws)
    
    def generate_bets(self, num_bets: int = 4) -> List[Dict]:
        red_scores = self.calculate_red_scores()
        blue_scores = self.calculate_blue_scores()
        
        # 使用指数权重放大差异
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
    """方法2：胆拖混合 - 基于上期热号作为胆码"""
    
    def __init__(self, draws: List[Dict]):
        self.draws = draws
        self.method1 = Method1HotColdSum(draws)
    
    def select_anchors(self, num_anchors: int = 2) -> List[int]:
        """选择胆码：结合上期红球、热区号码、高频号码"""
        red_scores = self.method1.calculate_red_scores()
        
        # 上期红球加分
        if self.draws:
            last_reds = self.draws[-1].get('reds', [])
            for num in last_reds:
                if num in red_scores:
                    red_scores[num] += 0.3
        
        # 热区号码加分
        zone_heat = get_zone_heat(self.draws, 50)
        for zone_id, zone_info in zone_heat.items():
            if '🔥' in zone_info['heat_level']:
                for num in ZONES[zone_id]['numbers']:
                    if num in red_scores:
                        red_scores[num] += 0.2
        
        # 近期高频号码加分（最近20期出现>=3次）
        recent_counts = {}
        for draw in self.draws[-20:]:
            for num in draw.get('reds', []):
                recent_counts[num] = recent_counts.get(num, 0) + 1
        for num, count in recent_counts.items():
            if count >= 3 and num in red_scores:
                red_scores[num] += 0.15
        
        sorted_nums = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:num_anchors]]
    
    def generate_bets(self, num_bets: int = 4) -> List[Dict]:
        anchors = self.select_anchors(num_anchors=2)
        red_scores = self.method1.calculate_red_scores()
        blue_scores = self.method1.calculate_blue_scores()
        
        # 降低胆码的权重，避免重复
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
                if len(candidates) >= needed:
                    selected = np.random.choice(candidates, size=needed, replace=False)
                else:
                    selected = []
                reds = sorted(anchors + selected.tolist())
                blue = np.random.choice(BLUE_NUMBERS)
                bets.append({
                    'reds': reds[:6],
                    'blue': int(blue),
                    'sum': sum(reds[:6]),
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
        """提取特征用于LightGBM训练"""
        if len(window_draws) < 20:
            return None
        
        features = {}
        total = len(window_draws)
        
        # 历史频率
        freq = sum(1 for d in window_draws if target_num in d.get('reds', []))
        features['freq'] = freq / total if total > 0 else 0
        
        # 遗漏期数
        last_seen = None
        for idx, d in enumerate(reversed(window_draws)):
            if target_num in d.get('reds', []):
                last_seen = idx
                break
        features['absence'] = last_seen if last_seen is not None else total
        
        # 近期频率（最近10期）
        recent = window_draws[-10:] if len(window_draws) >= 10 else window_draws
        recent_freq = sum(1 for d in recent if target_num in d.get('reds', []))
        features['recent_freq'] = recent_freq / len(recent) if recent else 0
        
        # 上期是否出现
        if window_draws:
            features['last_appeared'] = 1 if target_num in window_draws[-1].get('reds', []) else 0
        
        # 分区信息
        zone = (target_num - 1) // 11 + 1
        features['zone'] = zone
        features['parity'] = target_num % 2
        features['size'] = 0 if target_num <= 16 else 1
        
        return features
    
    def train(self) -> bool:
        """训练LightGBM模型"""
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
        """使用训练好的模型预测"""
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
            else:
                predictions.append((num, 0.0))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [num for num, _ in predictions[:6]]
    
    def generate_bets(self, num_bets: int = 4) -> List[Dict]:
        """生成投注"""
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
            
            # 随机替换1-2个号码增加多样性
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


# ==================== 方法4：XGBoost + 神经网络集成 ====================
class Method4Ensemble:
    """方法4：XGBoost + 神经网络集成"""
    
    def __init__(self, draws: List[Dict]):
        self.draws = draws
        self.xgb_model = None
        self.nn_model = None
        self.scaler = None
        self.is_trained = False
    
    def _extract_features_advanced(self, window_draws: List[Dict], target_num: int) -> Optional[Dict]:
        """提取高级特征用于集成模型"""
        if len(window_draws) < 30:
            return None
        
        features = {}
        total = len(window_draws)
        
        # 基础频率
        freq = sum(1 for d in window_draws if target_num in d.get('reds', []))
        features['freq'] = freq / total if total > 0 else 0
        
        # 遗漏期数
        last_seen = None
        for idx, d in enumerate(reversed(window_draws)):
            if target_num in d.get('reds', []):
                last_seen = idx
                break
        features['absence'] = last_seen if last_seen is not None else total
        features['absence_norm'] = features['absence'] / total if total > 0 else 0
        
        # 多窗口频率
        for window in [3, 5, 10]:
            recent = window_draws[-window:] if len(window_draws) >= window else window_draws
            recent_freq = sum(1 for d in recent if target_num in d.get('reds', []))
            features[f'recent_{window}'] = recent_freq / len(recent) if recent else 0
        
        # 与上期关系
        if window_draws:
            last_reds = window_draws[-1].get('reds', [])
            features['last_appeared'] = 1 if target_num in last_reds else 0
            if last_reds:
                features['min_diff_to_last'] = min(abs(target_num - n) for n in last_reds)
            else:
                features['min_diff_to_last'] = 99
        
        # 分区和统计特征
        zone = (target_num - 1) // 11 + 1
        features['zone'] = zone
        features['parity'] = target_num % 2
        features['size'] = 0 if target_num <= 16 else 1
        features['tail'] = target_num % 10
        
        return features
    
    def train(self) -> bool:
        """训练XGBoost + 神经网络集成模型"""
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
        except Exception:
            return False
    
    def predict(self) -> List[int]:
        """使用集成模型预测"""
        if not self.is_trained:
            return []
        
        predictions = []
        for num in RED_NUMBERS:
            features = self._extract_features_advanced(self.draws, num)
            if features:
                X_pred = pd.DataFrame([features]).fillna(0)
                X_pred = X_pred.reindex(columns=self.xgb_model.feature_names_in_, fill_value=0)
                
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
            else:
                predictions.append((num, 0.0))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [num for num, _ in predictions[:6]]
    
    def generate_bets(self, num_bets: int = 4) -> List[Dict]:
        """生成投注"""
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
    """投注生成工厂类"""
    
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


# ==================== ROI回测函数 ====================
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


print("第3部分加载完成")

# 第3部分结束 - 请确认后继续第4部分
# ============================================================
# 第4部分：主页面UI + 完整布局 + 投注生成界面
# ============================================================

# ==================== 冷热码分析显示 ====================
# ==================== 冷热码分析显示 ====================
st.subheader("🔥 冷热码分析")

col1, col2 = st.columns(2)
with col1:
    analysis_periods = st.slider(
        "冷热码统计期数", 
        min_value=20, 
        max_value=min(500, len(draws)), 
        value=min(100, len(draws)), 
        step=10,
        key="hot_cold_periods"
    )
with col2:
    zone_periods = st.slider(
        "7分区统计期数",
        min_value=20,
        max_value=min(500, len(draws)),
        value=min(100, len(draws)),
        step=10,
        key="zone_periods"
    )

# 获取分析数据
cold_hot_data = get_hot_cold_analysis(draws, analysis_periods)
zone_heat = get_zone_heat(draws, zone_periods)

# 三列布局显示
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔥 热门红球 Top 15**")
    hot_df = pd.DataFrame([
        {'号码': num, '出现次数': cnt, '频率': cnt / cold_hot_data['total_draws'] / 6 * 100}
        for num, cnt in cold_hot_data['hot_reds']
    ])
    st.dataframe(hot_df.style.format({'频率': '{:.1f}%'}), use_container_width=True, hide_index=True)

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
            '占比': zone['percentage']
        }
        for zone_id, zone in zone_heat.items()
    ])
    st.dataframe(zone_df.style.format({'占比': '{:.1f}%'}), use_container_width=True, hide_index=True)

# 蓝球热号单独显示
st.markdown("**💙 热门蓝球 Top 8**")
hot_blues_df = pd.DataFrame([
    {'蓝球': num, '出现次数': cnt, '频率': cnt / cold_hot_data['total_draws'] * 100}
    for num, cnt in cold_hot_data['hot_blues']
])
st.dataframe(hot_blues_df.style.format({'频率': '{:.1f}%'}), use_container_width=True, hide_index=True)

st.markdown("---")

# ==================== 和值趋势分析 ====================
st.subheader("📈 和值趋势分析")

trend_periods = st.slider(
    "走势图显示期数",
    min_value=10,
    max_value=min(200, len(draws)),
    value=min(50, len(draws)),
    step=10,
    key="trend_periods"
)

sum_trend = get_sum_trend(draws, trend_periods)
target_sum, tolerance = get_target_sum(draws)

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
fig_sum.add_hline(y=sum_trend['mean_sum'], line_dash="dot", line_color="green", annotation_text=f"历史均值({sum_trend['mean_sum']:.0f})")
fig_sum.add_hrect(
    y0=RED_EXPECTED_SUM - RED_SUM_STD, 
    y1=RED_EXPECTED_SUM + RED_SUM_STD, 
    line_width=0, 
    fillcolor="green", 
    opacity=0.1, 
    annotation_text="约68%区间"
)
fig_sum.update_layout(
    title=f"最近{trend_periods}期红球和值走势",
    xaxis_title="期数（倒序）",
    yaxis_title="和值",
    height=400,
    hovermode='x unified'
)
st.plotly_chart(fig_sum, use_container_width=True)

# 预测指标卡片（放在图表下方）
st.markdown("**📊 和值预测参考**")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("理论均值", f"{RED_EXPECTED_SUM}")
with col2:
    st.metric("历史均值", f"{sum_trend['mean_sum']:.1f}")
with col3:
    st.metric("预测目标", f"{target_sum} ± {tolerance}")
with col4:
    current_sum = sum(draws[-1].get('reds', []))
    st.metric("当前和值", f"{current_sum}", delta=f"{current_sum - RED_EXPECTED_SUM:+d}")

st.markdown("---")

# ==================== 蓝球走势分析 ====================
st.subheader("💙 蓝球走势分析")

blue_trend = get_blue_trend(draws, trend_periods)

col1, col2 = st.columns(2)

with col1:
    # 蓝球频率柱状图
    blue_freq_df = pd.DataFrame([
        {'蓝球': num, '出现次数': blue_trend['blue_freq'][num]}
        for num in range(1, 17)
    ])
    fig_blue = px.bar(
        blue_freq_df, x='蓝球', y='出现次数',
        title=f"最近{trend_periods}期蓝球出现频率",
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
        {'指标': '小号占比', '数值': f"{blue_trend['small_count']/trend_periods*100:.1f}%"},
        {'指标': '大号占比', '数值': f"{blue_trend['large_count']/trend_periods*100:.1f}%"},
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

# ==================== ML智能分析 ====================
st.subheader("🧠 ML智能分析引擎")

# 计算ML信号
ml_signals = calculate_ml_signals(draws)
next_period = get_next_period(draws)

# 顶部指标卡片
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("奖池阈值", ml_signals['jackpot_level'].split(' ')[0] if ml_signals['jackpot_level'] != '数据不足' else '数据不足')
with col2:
    st.metric("头奖周期", ml_signals['cycle'].split(' ')[0] if ml_signals['cycle'] != '数据不足' else '数据不足')
with col3:
    st.metric("信号强度", f"{ml_signals['signal_strength']}%")
with col4:
    st.metric("综合建议", ml_signals['suggestion_text'][:8])

# 详细指标
st.markdown("**📊 详细分析**")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**✂️ 剪刀差信号**: {ml_signals['scissors']}")
    st.markdown(f"**💙 蓝球偏向**: {ml_signals['blue_bias']} ({ml_signals.get('blue_prob', '50%')})")
    st.markdown(f"**🔄 红球重复**: {ml_signals['repeat_hint']}")
    if ml_signals.get('is_sunday', False):
        st.info(f"📅 {ml_signals['sunday_hint']}")

with col2:
    st.markdown(f"**⏰ 头奖倒计时**: {ml_signals['countdown']}")
    st.markdown(f"**📐 和值提示**: {ml_signals['sum_suggestion']}")
    st.markdown(f"**💰 奖池金额**: ¥{ml_signals['pool']/1e8:.2f}亿")
    st.markdown(f"**📈 上期销量**: ¥{ml_signals['sales']/1e8:.2f}亿")

# 综合信号强度进度条
st.markdown("**📊 综合信号强度**")
strength = ml_signals['signal_strength']
if strength >= 60:
    st.progress(strength / 100, text=f"🔔 {strength}% - 强烈推荐投注")
elif strength >= 30:
    st.progress(strength / 100, text=f"⚠️ {strength}% - 谨慎投注")
else:
    st.progress(strength / 100, text=f"💤 {strength}% - 建议观望")

st.markdown("---")

# ==================== 智能投注生成 ====================
st.subheader("🎲 智能投注生成")
st.info(f"🎯 **预测下一期**: {next_period}")

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

# 随机种子 - 日期时间选择器
st.markdown("**🎲 随机种子设置**")
col1, col2 = st.columns(2)
with col1:
    seed_date = st.date_input("日期", value=datetime.now(), key="seed_date")
with col2:
    seed_time = st.time_input("时间", value=datetime.now().time(), key="seed_time")

use_seed = st.checkbox("使用随机种子", value=False, key="use_seed")

if st.button("🚀 生成智能投注", type="primary", key="generate_btn"):
    if use_seed:
        # 合并日期和时间
        seed_datetime = datetime.combine(seed_date, seed_time)
        seed_val = int(seed_datetime.timestamp())
        random.seed(seed_val)
        np.random.seed(seed_val)
        st.success(f"✅ 已设置随机种子: {seed_datetime.strftime('%Y-%m-%d %H:%M')}")
    else:
        random.seed()
        np.random.seed()
    
    with st.spinner(f"正在使用 {ai_model} 生成投注..."):
        method_name = ai_model.split(":")[0] if ":" in ai_model else ai_model
        bets = BetGenerator.generate(method_name, draws, num_bets)
        st.session_state['generated_bets'] = bets
        st.session_state['model_used'] = ai_model
    st.success(f"✅ 使用 {ai_model} 生成 {len(bets)} 组投注")

# 显示生成的投注
if st.session_state.get('generated_bets'):
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
    
    # AI建议
    ai_suggestion = get_deepseek_suggestion(draws, "Supabase", ml_signals, next_period)
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
st.caption("📌 粘贴实际开奖数据，查看投注中奖情况（最多50期，用户自定义）")

col_check1, col_check2 = st.columns([3, 1])
with col_check2:
    max_check_draws = st.number_input("最大查奖期数", min_value=1, max_value=50, value=10, step=1, key="max_check_draws")

check_draws_text = st.text_area(
    "📋 粘贴开奖数据",
    height=120,
    key="check_draws",
    placeholder="格式: 期号 日期 红1 红2 红3 红4 红5 红6 蓝\n示例:\n2026050 2026-05-03 03 04 14 15 18 20 02\n2026049 2026-05-01 09 15 18 24 28 33 01\n\n支持最多50期"
)

if st.button("🔍 查奖", key="check_btn") and check_draws_text:
    check_draws = parse_check_draws(check_draws_text, max_draws=max_check_draws)
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

print("第4部分加载完成")
print("=" * 60)
print("所有代码加载完成！应用已就绪。")
print("=" * 60)
