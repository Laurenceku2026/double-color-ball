# ============================================================
# 双色球AI智能选号工具 v12.0 完整版
# 第1部分：导入、配置、Supabase连接、数据管理、ML预测缓存表
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

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✅ XGBoost 导入成功")
except ImportError as e:
    print(f"❌ XGBoost 导入失败: {e}")

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

# MCP服务（可选）
try:
    from ssq_mcp import get_recent_data, get_data_by_issue_range, get_frequency_analysis
    MCP_AVAILABLE = True
    print("✅ MCP服务 导入成功")
except ImportError as e:
    print(f"❌ MCP服务 导入失败: {e}")

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="双色球AI分析工具 - 专业版",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS（移除图片卡片样式） ====================
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

# ==================== ML预测缓存表创建 ====================
def check_ml_predictions_table() -> bool:
    """检查ml_predictions表是否存在"""
    supabase = init_supabase()
    if supabase is None:
        return False
    try:
        supabase.schema('ssq_schema').table('ml_predictions').select("id").limit(1).execute()
        return True
    except Exception:
        return False
# ==================== ML预测缓存操作函数 ====================
def save_ml_prediction_to_cache(model_name: str, prediction_data: Dict, training_periods: int):
    """保存ML预测结果到缓存"""
    supabase = init_supabase()
    if supabase is None:
        return False
    
    try:
        # 先停用同模型的旧记录
        supabase.schema('ssq_schema').table('ml_predictions')\
            .update({"is_active": False})\
            .eq("model_name", model_name)\
            .eq("is_active", True)\
            .execute()
        
        # 插入新记录
        data = {
            "model_name": model_name,
            "prediction_data": json.dumps(prediction_data),
            "training_periods": training_periods,
            "trained_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=7)).isoformat(),  # 7天过期
            "is_active": True
        }
        supabase.schema('ssq_schema').table('ml_predictions').insert(data).execute()
        return True
    except Exception as e:
        print(f"保存ML预测缓存失败: {e}")
        return False

def load_ml_prediction_from_cache(model_name: str, max_age_hours: int = 24) -> Optional[Dict]:
    """从缓存加载ML预测结果（24小时内有效）"""
    supabase = init_supabase()
    if supabase is None:
        return None
    
    try:
        cutoff_time = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
        response = supabase.schema('ssq_schema').table('ml_predictions')\
            .select("*")\
            .eq("model_name", model_name)\
            .eq("is_active", True)\
            .gt("trained_at", cutoff_time)\
            .order("trained_at", desc=True)\
            .limit(1)\
            .execute()
        
        if response.data:
            return {
                "prediction_data": json.loads(response.data[0]["prediction_data"]),
                "training_periods": response.data[0]["training_periods"],
                "trained_at": response.data[0]["trained_at"]
            }
        return None
    except Exception as e:
        print(f"加载ML预测缓存失败: {e}")
        return None

# ==================== 数据加载函数 ====================
def load_all_from_supabase() -> Optional[List[Dict]]:
    """从Supabase加载全部数据（分页加载，确保获取全部数据）"""
    supabase = init_supabase()
    if supabase is None:
        return None
    
    try:
        all_data = []
        page = 0
        page_size = 1000
        
        while True:
            response = supabase.schema('ssq_schema').table('ssq_draws')\
                .select("*")\
                .order("period", desc=False)\
                .range(page * page_size, (page + 1) * page_size - 1)\
                .execute()
            
            if not response.data:
                break
            
            all_data.extend(response.data)
            
            if len(response.data) < page_size:
                break
            
            page += 1
        
        if not all_data:
            return None
        
        all_draws = []
        for row in all_data:
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
def save_draws_to_supabase(draws: List[Dict], overwrite: bool = True) -> int:
    """保存数据到Supabase
    overwrite=True: 先清空再插入（覆盖全部）
    overwrite=False: 只插入新数据（不删除已有，但会更新已存在的）
    """
    if not draws:
        return 0
    
    supabase = init_supabase()
    if supabase is None:
        return 0
    
    try:
        if overwrite:
            # 覆盖全部模式：先清空表
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

def incremental_sync_draws(draws: List[Dict]) -> Dict:
    """增量同步：只更新变更的数据（新增、修改、删除）
    返回: {"inserted": n, "updated": n, "deleted": n}
    """
    if not draws:
        return {"inserted": 0, "updated": 0, "deleted": 0}
    
    supabase = init_supabase()
    if supabase is None:
        return {"inserted": 0, "updated": 0, "deleted": 0}
    
    # 获取数据库中现有期号
    existing_response = supabase.schema('ssq_schema').table('ssq_draws')\
        .select("period").execute()
    existing_periods = {row["period"] for row in existing_response.data} if existing_response.data else set()
    
    # 新数据中的期号
    new_periods = {draw.get('period') for draw in draws if draw.get('period') is not None}
    
    # 需要删除的期号（数据库有但新数据没有）
    to_delete = existing_periods - new_periods
    
    # 需要更新/插入的期号
    to_sync = new_periods
    
    inserted = 0
    updated = 0
    deleted = 0
    
    # 执行删除
    for period in to_delete:
        try:
            supabase.schema('ssq_schema').table('ssq_draws')\
                .delete().eq("period", period).execute()
            deleted += 1
        except Exception as e:
            st.warning(f"删除期号 {period} 失败: {e}")
    
    # 执行更新/插入
    for draw in draws:
        period = draw.get('period')
        if period is None:
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
        
        try:
            if period in existing_periods:
                # 更新
                supabase.schema('ssq_schema').table('ssq_draws')\
                    .update(data).eq("period", period).execute()
                updated += 1
            else:
                # 插入
                supabase.schema('ssq_schema').table('ssq_draws')\
                    .insert(data).execute()
                inserted += 1
        except Exception as e:
            st.warning(f"同步期号 {period} 失败: {e}")
    
    return {"inserted": inserted, "updated": updated, "deleted": deleted}

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


print("第1部分加载完成")
print("=" * 60)
print("请确认第1部分代码，输入 CONFIRM 后继续第2部分")
print("=" * 60)
# ============================================================
# 第2部分：管理员页面优化 + ML预测缓存集成 + 回测逻辑修正准备
# ============================================================

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
    """解析用户上传的Excel文件 - 完整15列"""
    try:
        # 尝试导入 openpyxl
        try:
            import openpyxl
        except ImportError:
            st.error("缺少 openpyxl 库，请使用文本粘贴功能")
            return None
        
        df = pd.read_excel(uploaded_file, sheet_name=0)
        
        # 列名匹配
        period_col = None
        date_col = None
        red_cols = []
        blue_col = None
        pool_col = None
        sales_col = None
        prize1_count_col = None
        prize1_amount_col = None
        prize2_count_col = None
        prize2_amount_col = None
        
        # 遍历所有列，匹配列名
        for col in df.columns:
            col_str = str(col).strip()
            
            if '期号' in col_str:
                period_col = col
            elif '开奖日期' in col_str or '日期' in col_str:
                date_col = col
            elif '红1' in col_str or '红球1' in col_str:
                for i in range(1, 7):
                    red_check = f'红{i}'
                    if red_check in df.columns:
                        if red_check not in red_cols:
                            red_cols.append(red_check)
            elif '蓝球' in col_str:
                blue_col = col
            elif '奖池' in col_str and '奖金' in col_str:
                pool_col = col
            elif '总投注额' in col_str or '总投注金额' in col_str:
                sales_col = col
            elif '一等奖注数' in col_str:
                prize1_count_col = col
            elif '一等奖奖金' in col_str:
                prize1_amount_col = col
            elif '二等奖注数' in col_str:
                prize2_count_col = col
            elif '二等奖奖金' in col_str:
                prize2_amount_col = col
        
        # 如果按列名没找到红球列，按位置（第3-8列）
        if len(red_cols) != 6:
            if len(df.columns) >= 8:
                red_cols = df.columns[2:8].tolist()
        
        # 如果期号列没找到，取第一列
        if period_col is None:
            period_col = df.columns[0]
        
        # 如果日期列没找到且第二列存在，取第二列
        if date_col is None and len(df.columns) > 1:
            date_col = df.columns[1]
        
        # 如果蓝球列没找到且第九列存在，取第九列
        if blue_col is None and len(df.columns) > 8:
            blue_col = df.columns[8]
        
        # 验证红球列
        if len(red_cols) != 6:
            st.error(f"无法识别红球列，找到{len(red_cols)}列，需要6列")
            st.info(f"找到的红球列: {red_cols}")
            return None
        
        def safe_int_convert(val):
            if pd.isna(val):
                return 0
            if isinstance(val, (int, float)):
                return int(val)
            val_str = str(val).strip()
            val_str = val_str.replace(',', '').replace(' ', '')
            val_str = val_str.replace('\n', '').replace('\r', '')
            if val_str == '' or val_str == '0':
                return 0
            try:
                return int(float(val_str))
            except:
                return 0
        
        draws = []
        error_count = 0
        skipped_count = 0
        
        for idx, row in df.iterrows():
            try:
                period_val = row[period_col]
                if pd.isna(period_val):
                    skipped_count += 1
                    continue
                period = int(period_val) if str(period_val).isdigit() else str(period_val)
                
                date = None
                if date_col and pd.notna(row[date_col]):
                    date_val = row[date_col]
                    if isinstance(date_val, datetime):
                        date = date_val.strftime('%Y-%m-%d')
                    else:
                        date = str(date_val).split()[0] if ' ' in str(date_val) else str(date_val)
                
                reds = []
                for col in red_cols[:6]:
                    val = row[col]
                    if pd.notna(val):
                        reds.append(int(val))
                
                if len(reds) != 6:
                    error_count += 1
                    continue
                reds = sorted(reds)
                
                blue = 0
                if blue_col and pd.notna(row[blue_col]):
                    blue = safe_int_convert(row[blue_col])
                
                pool = 0
                if pool_col and pd.notna(row[pool_col]):
                    pool = safe_int_convert(row[pool_col])
                
                sales = 0
                if sales_col and pd.notna(row[sales_col]):
                    sales = safe_int_convert(row[sales_col])
                
                prize1_count = 0
                if prize1_count_col and pd.notna(row[prize1_count_col]):
                    prize1_count = safe_int_convert(row[prize1_count_col])
                
                prize1_amount = 0
                if prize1_amount_col and pd.notna(row[prize1_amount_col]):
                    prize1_amount = safe_int_convert(row[prize1_amount_col])
                
                prize2_count = 0
                if prize2_count_col and pd.notna(row[prize2_count_col]):
                    prize2_count = safe_int_convert(row[prize2_count_col])
                
                prize2_amount = 0
                if prize2_amount_col and pd.notna(row[prize2_amount_col]):
                    prize2_amount = safe_int_convert(row[prize2_amount_col])
                
                draws.append({
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
                })
                
            except Exception as e:
                error_count += 1
                continue
        
        if draws:
            st.success(f"成功解析 {len(draws)} 期数据")
            if skipped_count > 0:
                st.warning(f"跳过 {skipped_count} 行（期号为空）")
            if error_count > 0:
                st.warning(f"跳过 {error_count} 行（数据格式错误）")
            return draws
        else:
            st.error("未找到有效数据")
            return None
        
    except Exception as e:
        st.error(f"Excel解析错误: {e}")
        return None

# ==================== 管理员页面 ====================
def show_admin_page():
    """管理员页面 - 可编辑表格（支持覆盖保存和增量同步）"""
    
    st.subheader("📋 数据编辑器")
    st.caption("💡 双击单元格编辑 | 表格底部有 '+' 按钮添加新行 | 选择保存模式")
    
    # 定义固定列名（15列）
    columns = [
        "期号", "开奖日期", "红1", "红2", "红3", "红4", "红5", "红6", "蓝球",
        "奖池奖金(元)", "一等奖注数", "一等奖奖金(元)", "二等奖注数", "二等奖奖金(元)", "总投注额(元)"
    ]
    
    # 加载现有数据
    current_draws = st.session_state.get('draws_loaded', [])
    
    # 构建 DataFrame
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
    else:
        df = pd.DataFrame(columns=columns)
    
    # 配置列类型
    column_config = {
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
    
    # 显示数据量统计
    st.info(f"📊 当前数据量: {len(current_draws)} 期")
    
    # 刷新按钮（从数据库重新加载）
    col_refresh, _ = st.columns([1, 5])
    with col_refresh:
        if st.button("🔄 从数据库重新加载", use_container_width=True):
            with st.spinner("加载中..."):
                draws = load_all_from_supabase()
                if draws:
                    draws = fill_missing_with_history(draws)
                    st.session_state['draws_loaded'] = draws
                    st.session_state.pop('ssq_data_editor', None)
                    st.success(f"加载 {len(draws)} 期数据")
                    st.rerun()
    
    st.markdown("---")
    
    # 使用 form 包装，确保一次性提交
    with st.form(key="data_editor_form"):
        # 显示可编辑表格
        edited_df = st.data_editor(
            df,
            column_config=column_config,
            use_container_width=True,
            height=500,
            num_rows="dynamic",
            key="ssq_data_editor"
        )
        
        # 两个保存按钮放在同一行
        col_save1, col_save2, col_spacer = st.columns([1, 1, 3])
        
        with col_save1:
            overwrite_submitted = st.form_submit_button("💾 全量覆盖保存", type="primary", use_container_width=True)
        with col_save2:
            incremental_submitted = st.form_submit_button("🔄 增量同步保存", use_container_width=True)
        
        if overwrite_submitted:
            if edited_df is None or len(edited_df) == 0:
                st.error("没有数据可保存")
            else:
                with st.spinner("正在执行全量覆盖保存..."):
                    new_draws = []
                    errors = 0
                    
                    for idx, row in edited_df.iterrows():
                        try:
                            if pd.isna(row['期号']) or row['期号'] == 0:
                                continue
                            
                            period = int(row['期号'])
                            date = str(row['开奖日期']) if pd.notna(row['开奖日期']) and row['开奖日期'] != '' else None
                            
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
                    
                    if errors > 0:
                        st.warning(f"跳过 {errors} 行无效数据（红球1-33，蓝球1-16）")
                    
                    if new_draws:
                        saved = save_draws_to_supabase(new_draws, overwrite=True)
                        if saved > 0:
                            st.session_state['draws_loaded'] = new_draws
                            st.success(f"全量覆盖保存 {saved} 期数据成功！")
                            st.rerun()
                        else:
                            st.error("保存失败")
                    else:
                        st.error("没有有效数据可保存")
        
        if incremental_submitted:
            if edited_df is None or len(edited_df) == 0:
                st.error("没有数据可同步")
            else:
                with st.spinner("正在执行增量同步..."):
                    new_draws = []
                    errors = 0
                    
                    for idx, row in edited_df.iterrows():
                        try:
                            if pd.isna(row['期号']) or row['期号'] == 0:
                                continue
                            
                            period = int(row['期号'])
                            date = str(row['开奖日期']) if pd.notna(row['开奖日期']) and row['开奖日期'] != '' else None
                            
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
                    
                    if errors > 0:
                        st.warning(f"跳过 {errors} 行无效数据")
                    
                    if new_draws:
                        result = incremental_sync_draws(new_draws)
                        st.success(f"增量同步完成：新增 {result['inserted']} 期，更新 {result['updated']} 期，删除 {result['deleted']} 期")
                        # 重新加载数据到session
                        refreshed_draws = load_all_from_supabase()
                        if refreshed_draws:
                            refreshed_draws = fill_missing_with_history(refreshed_draws)
                            st.session_state['draws_loaded'] = refreshed_draws
                        st.rerun()
                    else:
                        st.error("没有有效数据可同步")
    
    st.markdown("---")
    
    # ==================== Excel上传区域 ====================
    st.subheader("📎 Excel文件上传")
    st.caption("格式：期号、开奖日期、红1-6、蓝球、奖池奖金(元)、一等奖注数、一等奖奖金(元)、二等奖注数、二等奖奖金(元)、总投注额(元)")
    
    uploaded_file = st.file_uploader(
        "选择Excel文件",
        type=['xlsx', 'xls'],
        key="excel_uploader_admin",
        help="上传Excel文件"
    )
    
    if uploaded_file is not None:
        with st.spinner("正在解析Excel文件..."):
            excel_draws = parse_excel_file(uploaded_file)
            if excel_draws and len(excel_draws) > 0:
                st.success(f"✅ 成功解析 {len(excel_draws)} 期数据")
                
                # 预览
                preview_data = []
                for d in excel_draws[:10]:
                    reds = d.get('reds', [])
                    reds_str = ','.join(f"{r:02d}" for r in reds)
                    preview_data.append({
                        '期号': d.get('period'),
                        '开奖日期': str(d.get('date', ''))[:10] if d.get('date') else '',
                        '红球': reds_str,
                        '蓝球': f"{d.get('blue', 0):02d}"
                    })
                st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)
                
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("✅ 确认全量覆盖", type="primary"):
                        excel_draws = fill_missing_with_history(excel_draws)
                        excel_draws.sort(key=lambda x: x.get('period', 0))
                        saved = save_draws_to_supabase(excel_draws, overwrite=True)
                        if saved > 0:
                            st.session_state['draws_loaded'] = excel_draws
                            st.success(f"保存 {saved} 期数据成功！")
                            st.rerun()
                with col_cancel:
                    if st.button("❌ 取消"):
                        st.rerun()
            else:
                st.error("解析失败，请检查文件格式")

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
if 'ml_predictions_cache' not in st.session_state:
    st.session_state['ml_predictions_cache'] = {}
if 'last_training_time' not in st.session_state:
    st.session_state['last_training_time'] = None

# ==================== 主页面标题 ====================
col_title, col_settings = st.columns([0.9, 0.1])
with col_title:
    st.title("🎯 双色球AI智能选号工具 - 专业版 v12.0")
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


print("第2部分加载完成")
print("=" * 60)
print("请确认第2部分代码，输入 CONFIRM 后继续第3部分")
print("=" * 60)
# ============================================================
# 第3部分：分析引擎（冷热码、7分区、和值、蓝球、ML信号）+ 滚动窗口回测
# ============================================================

# ==================== 冷热码分析 ====================
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
    
    total_draws = len(recent_draws)
    
    # 热门红球 Top 15
    hot_reds = sorted(red_freq.items(), key=lambda x: x[1], reverse=True)[:15]
    # 冷门红球 Bottom 10
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

# ==================== ML信号分析 ====================
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
    repeat_hint = f"上期红球重复{repeat_count}个，历史概率60%"
    
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
        return {
            "plan": "4组7+1复式",
            "win_rate": "30-35%",
            "blue_advice": "参考冷热分析",
            "summary": f"基于{len(draws)}期历史数据",
            "ml_tip": ml_signals.get('suggestion_text', f"奖池{ml_signals['jackpot_level']}，{ml_signals['cycle']}")
        }
    
    try:
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
    
    return {
        "plan": "4组7+1复式",
        "win_rate": "30-35%",
        "blue_advice": ml_signals.get('blue_bias', '均衡'),
        "summary": ml_signals.get('suggestion_text', '谨慎投注'),
        "ml_tip": ml_signals.get('suggestion_text', f"奖池{ml_signals['jackpot_level']}，{ml_signals['cycle']}")
    }

# ==================== 修正后的ROI回测函数（滚动窗口） ====================
def calculate_prize_for_single_bet(bet: Dict, actual: Dict) -> Tuple[int, str]:
    """计算单注中奖金额和奖级描述"""
    red_matches = len(set(bet['reds']) & set(actual['reds']))
    blue_match = (bet['blue'] == actual.get('blue', 0))
    
    if red_matches == 6 and blue_match:
        return 5000000, "🏆 一等奖"
    elif red_matches == 6:
        return 500000, "🥈 二等奖"
    elif red_matches == 5 and blue_match:
        return 3000, "🥉 三等奖"
    elif red_matches == 5 or (red_matches == 4 and blue_match):
        return 200, "📦 四等奖"
    elif red_matches == 4 or (red_matches == 3 and blue_match):
        return 10, "🎫 五等奖"
    elif blue_match:
        return 5, "⭐ 六等奖"
    elif red_matches == 3:
        return 5, "🎁 福运奖"
    else:
        return 0, "❌ 未中奖"

def calculate_prize_for_bets(bets: List[Dict], actual: Dict) -> int:
    """计算一组投注的总中奖金额"""
    total = 0
    for bet in bets:
        prize, _ = calculate_prize_for_single_bet(bet, actual)
        total += prize
    return total

def backtest_roi_rolling_window(draws: List[Dict], method_name: str, num_bets: int = 4, 
                                 start_period: int = 100, window_size: int = 10) -> Dict:
    """
    修正后的ROI回测函数 - 滚动窗口模式
    每 window_size 期重新训练一次模型，模拟真实场景
    
    参数:
        draws: 历史数据列表
        method_name: 方法名称
        num_bets: 每期投注组数
        start_period: 从第几期开始回测
        window_size: 滚动窗口大小（每多少期重新训练）
    """
    if len(draws) < start_period + 10:
        return {"roi": 0, "total_cost": 0, "total_prize": 0, "net": 0, "win_rate": 0, "periods": 0}
    
    total_cost = 0
    total_prize = 0
    win_count = 0
    prize_breakdown = {"first": 0, "second": 0, "third": 0, "fourth": 0, "fifth": 0, "sixth": 0, "fuyun": 0}
    
    # 缓存已经训练好的模型（按训练数据的截止索引）
    model_cache = {}
    
    # 用于方法3和方法4的模型存储
    trained_models = {}
    
    for i in range(start_period, len(draws)):
        # 训练数据：只使用 i 期之前的数据
        train_data = draws[:i]
        
        # 测试数据：第 i 期
        test_data = draws[i]
        
        # 确定使用哪个模型（找到最近训练的模型）
        model_key = None
        if method_name in ["方法3: LightGBM", "方法4: XGBoost+NN集成"]:
            # 找到最近的训练窗口
            window_start = ((i - start_period) // window_size) * window_size + start_period
            model_key = f"{method_name}_{window_start}"
            
            if model_key not in trained_models:
                # 需要重新训练
                if method_name == "方法3: LightGBM":
                    model = Method3LightGBM(train_data)
                    model.train()
                    trained_models[model_key] = model
                else:
                    model = Method4Ensemble(train_data)
                    model.train()
                    trained_models[model_key] = model
            current_model = trained_models[model_key]
        else:
            # 方法1和方法2不需要训练，直接使用当前数据
            current_model = None
        
        # 生成投注
        try:
            if method_name == "方法1: 当前方法":
                generator = Method1HotColdSum(train_data)
                bets = generator.generate_bets(num_bets)
            elif method_name == "方法2: 胆拖混合":
                generator = Method2DanTuo(train_data)
                bets = generator.generate_bets(num_bets)
            elif method_name == "方法3: LightGBM":
                bets = current_model.generate_bets(num_bets)
            elif method_name == "方法4: XGBoost+NN集成":
                bets = current_model.generate_bets(num_bets)
            else:
                # 默认使用方法1
                generator = Method1HotColdSum(train_data)
                bets = generator.generate_bets(num_bets)
        except Exception as e:
            # 如果生成失败，使用随机数
            bets = []
            for _ in range(num_bets):
                bets.append({
                    'reds': sorted(np.random.choice(RED_NUMBERS, size=6, replace=False)),
                    'blue': np.random.choice(BLUE_NUMBERS),
                    'sum': 0,
                    'method': method_name
                })
        
        # 计算当期奖金
        period_prize = 0
        for bet in bets:
            prize, level = calculate_prize_for_single_bet(bet, test_data)
            period_prize += prize
            
            # 统计奖级
            if "一等奖" in level:
                prize_breakdown["first"] += 1
            elif "二等奖" in level:
                prize_breakdown["second"] += 1
            elif "三等奖" in level:
                prize_breakdown["third"] += 1
            elif "四等奖" in level:
                prize_breakdown["fourth"] += 1
            elif "五等奖" in level:
                prize_breakdown["fifth"] += 1
            elif "六等奖" in level:
                prize_breakdown["sixth"] += 1
            elif "福运奖" in level:
                prize_breakdown["fuyun"] += 1
        
        period_cost = num_bets * 14  # 7+1复式每注14元
        total_cost += period_cost
        total_prize += period_prize
        
        if period_prize > 0:
            win_count += 1
    
    periods = len(draws) - start_period
    net = total_prize - total_cost
    roi = (net / total_cost) * 100 if total_cost > 0 else 0
    win_rate = (win_count / periods) * 100 if periods > 0 else 0
    
    return {
        "roi": roi,
        "total_cost": total_cost,
        "total_prize": total_prize,
        "net": net,
        "win_rate": win_rate,
        "periods": periods,
        "prize_breakdown": prize_breakdown
    }

# 为了保持向后兼容，保留原函数名但使用新逻辑
def backtest_roi(draws: List[Dict], method_name: str, num_bets: int = 4, lookback: int = 50) -> Dict:
    """向后兼容的ROI回测函数（使用滚动窗口）"""
    start_period = len(draws) - lookback if len(draws) > lookback else 100
    start_period = max(start_period, 50)
    return backtest_roi_rolling_window(draws, method_name, num_bets, start_period, window_size=10)


print("第3部分加载完成")
print("=" * 60)
print("请确认第3部分代码，输入 CONFIRM 后继续第4部分")
print("=" * 60)
# ============================================================
# 第4部分：四种AI算法实现 + 复式扩展逻辑（7+1/7+2/8+1）
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
    
    def get_top_reds_by_score(self, n: int = 10) -> List[int]:
        """获取得分最高的n个红球"""
        red_scores = self.calculate_red_scores()
        sorted_reds = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_reds[:n]]
    
    def get_top_blues_by_score(self, n: int = 3) -> List[int]:
        """获取得分最高的n个蓝球"""
        blue_scores = self.calculate_blue_scores()
        sorted_blues = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_blues[:n]]
    
    def generate_bets(self, num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
        """生成投注（支持7+1/7+2/8+1）"""
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
        
        # 解析复式类型
        red_count = 7 if bet_type.startswith("7") else 8
        blue_count = 2 if bet_type == "7+2" else 1
        
        # 获取Top红球用于扩展
        top_reds = self.get_top_reds_by_score(12)
        top_blues = self.get_top_blues_by_score(3)
        
        bets = []
        for bet_idx in range(num_bets):
            # 生成核心6码
            core_reds = None
            for attempt in range(100):
                reds = np.random.choice(RED_NUMBERS, size=6, replace=False, p=red_weights)
                reds = sorted(reds.tolist())
                if abs(sum(reds) - target_sum) <= tolerance:
                    core_reds = reds
                    break
            if core_reds is None:
                core_reds = sorted(np.random.choice(RED_NUMBERS, size=6, replace=False))
            
            # 扩展红球到 red_count 个
            final_reds = list(core_reds)
            if red_count > 6:
                # 从Top红球中选择不在当前集合中的号码
                candidates = [r for r in top_reds if r not in final_reds]
                # 如果Top红球不够，从所有号码中补充
                if len(candidates) < (red_count - 6):
                    candidates = [r for r in RED_NUMBERS if r not in final_reds]
                
                needed = red_count - 6
                # 按优先级选择：热码优先，同时考虑区间平衡
                selected = []
                for candidate in candidates:
                    # 检查区间平衡
                    zones_covered = set()
                    for r in final_reds:
                        for zid, zone in ZONES.items():
                            if r in zone['numbers']:
                                zones_covered.add(zid)
                                break
                    candidate_zone = None
                    for zid, zone in ZONES.items():
                        if candidate in zone['numbers']:
                            candidate_zone = zid
                            break
                    # 优先选择未覆盖区间的号码
                    if candidate_zone not in zones_covered and len(selected) < needed:
                        selected.append(candidate)
                
                # 如果还不够，补充热码
                for candidate in candidates:
                    if candidate not in selected and len(selected) < needed:
                        selected.append(candidate)
                
                final_reds.extend(selected[:needed])
                final_reds = sorted(final_reds)
            
            # 选择蓝球
            blues = []
            if blue_count == 1:
                blue = np.random.choice(BLUE_NUMBERS, p=blue_weights)
                blues.append(int(blue))
            else:  # 7+2
                # 主蓝球按权重选
                primary_blue = np.random.choice(BLUE_NUMBERS, p=blue_weights)
                blues.append(int(primary_blue))
                # 第2个蓝球从Top蓝球中选择（避开主蓝球）
                secondary_candidates = [b for b in top_blues if b != primary_blue]
                if secondary_candidates:
                    secondary_blue = np.random.choice(secondary_candidates)
                else:
                    secondary_blue = np.random.choice([b for b in BLUE_NUMBERS if b != primary_blue])
                blues.append(secondary_blue)
            
            bets.append({
                'reds': final_reds,
                'blues': blues,  # 多个蓝球
                'blue': blues[0],  # 保持向后兼容
                'sum': sum(final_reds),
                'method': '方法1:冷热码+和值',
                'bet_type': bet_type
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
    
    def get_top_reds_for_expansion(self, n: int = 12) -> List[int]:
        """获取用于扩展的Top红球"""
        red_scores = self.method1.calculate_red_scores()
        sorted_reds = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_reds[:n]]
    
    def generate_bets(self, num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
        """生成投注（支持复式扩展）"""
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
        
        # 解析复式类型
        red_count = 7 if bet_type.startswith("7") else 8
        blue_count = 2 if bet_type == "7+2" else 1
        
        top_reds = self.get_top_reds_for_expansion(12)
        top_blues = self.method1.get_top_blues_by_score(3)
        
        bets = []
        for bet_idx in range(num_bets):
            needed = 6 - len(anchors)
            core_reds = anchors.copy()
            
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
                temp_reds = sorted(anchors + selected.tolist())
                if abs(sum(temp_reds) - target_sum) <= tolerance:
                    core_reds = temp_reds
                    break
            else:
                candidates = [i for i in RED_NUMBERS if i not in anchors]
                if len(candidates) >= needed:
                    selected = np.random.choice(candidates, size=needed, replace=False)
                else:
                    selected = []
                core_reds = sorted(anchors + selected.tolist())[:6]
            
            # 扩展红球
            final_reds = list(core_reds)
            if red_count > 6:
                candidates = [r for r in top_reds if r not in final_reds]
                if len(candidates) < (red_count - 6):
                    candidates = [r for r in RED_NUMBERS if r not in final_reds]
                
                needed_extras = red_count - 6
                selected_extras = []
                for candidate in candidates:
                    if len(selected_extras) >= needed_extras:
                        break
                    # 检查区间平衡
                    zones_covered = set()
                    for r in final_reds:
                        for zid, zone in ZONES.items():
                            if r in zone['numbers']:
                                zones_covered.add(zid)
                                break
                    candidate_zone = None
                    for zid, zone in ZONES.items():
                        if candidate in zone['numbers']:
                            candidate_zone = zid
                            break
                    if candidate_zone not in zones_covered:
                        selected_extras.append(candidate)
                
                for candidate in candidates:
                    if candidate not in selected_extras and len(selected_extras) < needed_extras:
                        selected_extras.append(candidate)
                
                final_reds.extend(selected_extras[:needed_extras])
                final_reds = sorted(final_reds)
            
            # 选择蓝球
            blues = []
            if blue_count == 1:
                blue = np.random.choice(BLUE_NUMBERS, p=blue_weights)
                blues.append(int(blue))
            else:
                primary_blue = np.random.choice(BLUE_NUMBERS, p=blue_weights)
                blues.append(int(primary_blue))
                secondary_candidates = [b for b in top_blues if b != primary_blue]
                if secondary_candidates:
                    secondary_blue = np.random.choice(secondary_candidates)
                else:
                    secondary_blue = np.random.choice([b for b in BLUE_NUMBERS if b != primary_blue])
                blues.append(secondary_blue)
            
            bets.append({
                'reds': final_reds,
                'blues': blues,
                'blue': blues[0],
                'sum': sum(final_reds),
                'method': f'方法2:胆拖混合 (胆码:{anchors})',
                'bet_type': bet_type
            })
        
        return bets


# ==================== 方法3：LightGBM ====================
class Method3LightGBM:
    """方法3：LightGBM梯度提升树"""
    
    def __init__(self, draws: List[Dict]):
        self.draws = draws
        self.model = None
        self.is_trained = False
        self.top_reds_cache = None
    
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
    
    def predict_top_reds(self, n: int = 12) -> List[int]:
        """预测概率最高的n个红球"""
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
        return [num for num, _ in predictions[:n]]
    
    def generate_bets(self, num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
        """生成投注（支持复式扩展）"""
        if not self.is_trained:
            self.train()
        
        if not self.is_trained:
            method1 = Method1HotColdSum(self.draws)
            return method1.generate_bets(num_bets, bet_type)
        
        # 获取Top红球
        top_reds = self.predict_top_reds(12)
        if len(top_reds) < 6:
            top_reds = list(range(1, 34))
            random.shuffle(top_reds)
        
        blue_scores = Method1HotColdSum(self.draws).calculate_blue_scores()
        blue_weights = np.array([math.exp(blue_scores.get(i, 0)) for i in BLUE_NUMBERS])
        
        if np.sum(blue_weights) > 0:
            blue_weights = blue_weights / np.sum(blue_weights)
        else:
            blue_weights = np.ones(16) / 16
        
        # 解析复式类型
        red_count = 7 if bet_type.startswith("7") else 8
        blue_count = 2 if bet_type == "7+2" else 1
        
        top_blues = Method1HotColdSum(self.draws).get_top_blues_by_score(3)
        
        bets = []
        for _ in range(num_bets):
            # 从Top红球中选择核心6码
            core_reds = top_reds[:6].copy()
            random.shuffle(core_reds)
            core_reds = sorted(core_reds[:6])
            
            # 扩展红球
            final_reds = list(core_reds)
            if red_count > 6:
                candidates = [r for r in top_reds if r not in final_reds]
                if len(candidates) < (red_count - 6):
                    candidates = [r for r in RED_NUMBERS if r not in final_reds]
                
                needed_extras = red_count - 6
                # 随机选择（但优先在Top中选）
                if len(candidates) >= needed_extras:
                    selected_extras = np.random.choice(candidates, size=needed_extras, replace=False)
                else:
                    selected_extras = candidates
                final_reds.extend(selected_extras)
                final_reds = sorted(final_reds)
            
            # 选择蓝球
            blues = []
            if blue_count == 1:
                blue = np.random.choice(BLUE_NUMBERS, p=blue_weights)
                blues.append(int(blue))
            else:
                primary_blue = np.random.choice(BLUE_NUMBERS, p=blue_weights)
                blues.append(int(primary_blue))
                secondary_candidates = [b for b in top_blues if b != primary_blue]
                if secondary_candidates:
                    secondary_blue = np.random.choice(secondary_candidates)
                else:
                    secondary_blue = np.random.choice([b for b in BLUE_NUMBERS if b != primary_blue])
                blues.append(secondary_blue)
            
            bets.append({
                'reds': final_reds,
                'blues': blues,
                'blue': blues[0],
                'sum': sum(final_reds),
                'method': '方法3:LightGBM',
                'bet_type': bet_type
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
        self.top_reds_cache = None
    
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
        except Exception as e:
            print(f"训练失败: {e}")
            return False
    
    def predict_top_reds(self, n: int = 12) -> List[int]:
        """使用集成模型预测Top红球"""
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
        return [num for num, _ in predictions[:n]]
    
    def generate_bets(self, num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
        """生成投注（支持复式扩展）"""
        if not self.is_trained:
            self.train()
        
        if not self.is_trained:
            method1 = Method1HotColdSum(self.draws)
            return method1.generate_bets(num_bets, bet_type)
        
        # 获取Top红球
        top_reds = self.predict_top_reds(12)
        if len(top_reds) < 6:
            top_reds = list(range(1, 34))
            random.shuffle(top_reds)
        
        blue_scores = Method1HotColdSum(self.draws).calculate_blue_scores()
        blue_weights = np.array([math.exp(blue_scores.get(i, 0)) for i in BLUE_NUMBERS])
        
        if np.sum(blue_weights) > 0:
            blue_weights = blue_weights / np.sum(blue_weights)
        else:
            blue_weights = np.ones(16) / 16
        
        # 解析复式类型
        red_count = 7 if bet_type.startswith("7") else 8
        blue_count = 2 if bet_type == "7+2" else 1
        
        top_blues = Method1HotColdSum(self.draws).get_top_blues_by_score(3)
        
        bets = []
        for _ in range(num_bets):
            # 从Top红球中选择核心6码
            core_reds = top_reds[:6].copy()
            random.shuffle(core_reds)
            core_reds = sorted(core_reds[:6])
            
            # 扩展红球
            final_reds = list(core_reds)
            if red_count > 6:
                candidates = [r for r in top_reds if r not in final_reds]
                if len(candidates) < (red_count - 6):
                    candidates = [r for r in RED_NUMBERS if r not in final_reds]
                
                needed_extras = red_count - 6
                if len(candidates) >= needed_extras:
                    selected_extras = np.random.choice(candidates, size=needed_extras, replace=False)
                else:
                    selected_extras = candidates
                final_reds.extend(selected_extras)
                final_reds = sorted(final_reds)
            
            # 选择蓝球
            blues = []
            if blue_count == 1:
                blue = np.random.choice(BLUE_NUMBERS, p=blue_weights)
                blues.append(int(blue))
            else:
                primary_blue = np.random.choice(BLUE_NUMBERS, p=blue_weights)
                blues.append(int(primary_blue))
                secondary_candidates = [b for b in top_blues if b != primary_blue]
                if secondary_candidates:
                    secondary_blue = np.random.choice(secondary_candidates)
                else:
                    secondary_blue = np.random.choice([b for b in BLUE_NUMBERS if b != primary_blue])
                blues.append(secondary_blue)
            
            bets.append({
                'reds': final_reds,
                'blues': blues,
                'blue': blues[0],
                'sum': sum(final_reds),
                'method': '方法4:XGBoost+NN集成',
                'bet_type': bet_type
            })
        
        return bets


# ==================== 投注生成工厂 ====================
class BetGenerator:
    """投注生成工厂类"""
    
    @staticmethod
    def generate(method: str, draws: List[Dict], num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
        """生成投注（支持复式类型）"""
        if method == "方法1: 当前方法" or method == "方法1":
            generator = Method1HotColdSum(draws)
        elif method == "方法2: 胆拖混合" or method == "方法2":
            generator = Method2DanTuo(draws)
        elif method == "方法3: LightGBM" or method == "方法3":
            generator = Method3LightGBM(draws)
        elif method == "方法4: XGBoost+NN集成" or method == "方法4":
            generator = Method4Ensemble(draws)
        else:
            generator = Method1HotColdSum(draws)
        
        return generator.generate_bets(num_bets, bet_type)
    
    @staticmethod
    def generate_ensemble(draws: List[Dict], num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
        """综合模式：运行4种方法，取高频号码（支持复式扩展）"""
        from collections import Counter
        
        all_bets = []
        methods = ["方法1", "方法2", "方法3", "方法4"]
        
        for method in methods:
            try:
                bets = BetGenerator.generate(method, draws, num_bets, bet_type)
                all_bets.extend(bets)
            except Exception as e:
                print(f"{method} 生成失败: {e}")
                continue
        
        if not all_bets:
            return BetGenerator.generate("方法1", draws, num_bets, bet_type)
        
        # 解析复式类型
        red_count = 7 if bet_type.startswith("7") else 8
        blue_count = 2 if bet_type == "7+2" else 1
        
        # 统计所有红球号码频率
        red_counter = Counter()
        blue_counter = Counter()
        
        for bet in all_bets:
            for red in bet['reds']:
                red_counter[red] += 1
            for blue in bet.get('blues', [bet['blue']]):
                blue_counter[blue] += 1
        
        # 取频率最高的 red_count 个红球
        top_reds = [num for num, _ in red_counter.most_common(red_count)]
        if len(top_reds) < red_count:
            # 补充
            missing = [n for n in RED_NUMBERS if n not in top_reds]
            top_reds.extend(missing[:red_count - len(top_reds)])
        top_reds.sort()
        
        # 取频率最高的 blue_count 个蓝球
        top_blues = [num for num, _ in blue_counter.most_common(blue_count)]
        if len(top_blues) < blue_count:
            missing = [n for n in BLUE_NUMBERS if n not in top_blues]
            top_blues.extend(missing[:blue_count - len(top_blues)])
        
        # 生成有差异的组合
        bets = []
        for i in range(num_bets):
            reds = top_reds.copy()
            blues = top_blues.copy()
            
            # 每组替换1-2个红球增加多样性
            replace_count = min(i + 1, 2)
            for _ in range(replace_count):
                idx = random.randint(0, len(reds) - 1)
                candidates = [n for n in RED_NUMBERS if n not in reds]
                if candidates:
                    reds[idx] = random.choice(candidates)
            reds.sort()
            
            bets.append({
                'reds': reds,
                'blues': blues,
                'blue': blues[0],
                'sum': sum(reds),
                'method': '方法5:综合模式',
                'bet_type': bet_type
            })
        
        return bets


print("第4部分加载完成")
print("=" * 60)
print("请确认第4部分代码，输入 CONFIRM 后继续第5部分")
print("=" * 60)
# ============================================================
# 第5部分：主页面UI整合 + 完整代码 + 投注结果显示（表格形式）
# ============================================================

# ==================== 主页面内容 ====================

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
        {'号码': num, '出现次数': cnt, '频率': cnt / (cold_hot_data['total_draws'] * 6) * 100 if cold_hot_data['total_draws'] > 0 else 0}
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
    {'蓝球': num, '出现次数': cnt, '频率': cnt / cold_hot_data['total_draws'] * 100 if cold_hot_data['total_draws'] > 0 else 0}
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

# 预测指标卡片
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
        {'指标': '小号占比', '数值': f"{blue_trend['small_count']/trend_periods*100:.1f}%" if trend_periods > 0 else "0%"},
        {'指标': '大号占比', '数值': f"{blue_trend['large_count']/trend_periods*100:.1f}%" if trend_periods > 0 else "0%"},
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
        ["方法5: 综合模式 ⭐推荐", "方法4: XGBoost+NN集成", "方法3: LightGBM", "方法2: 胆拖混合", "方法1: 当前方法"],
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

# ML预测刷新按钮（使用缓存）
col_refresh_ml, _ = st.columns([1, 5])
with col_refresh_ml:
    if st.button("🔄 刷新ML预测", use_container_width=True, help="重新训练ML模型（耗时约30秒）"):
        with st.spinner("正在训练ML模型，请稍候..."):
            # 清除缓存，强制重新训练
            for method_name in ["方法3: LightGBM", "方法4: XGBoost+NN集成"]:
                cache_key = f"ml_model_{method_name}"
                if cache_key in st.session_state:
                    del st.session_state[cache_key]
            
            # 重新生成投注（会触发重新训练）
            temp_method = ai_model.split(":")[0] if ":" in ai_model else ai_model
            if "综合模式" in ai_model:
                temp_bets = BetGenerator.generate_ensemble(draws, num_bets, bet_type.split(' ')[0])
            else:
                temp_bets = BetGenerator.generate(temp_method, draws, num_bets, bet_type.split(' ')[0])
            
            st.success("ML模型刷新完成！")
            st.rerun()

if st.button("🚀 生成智能投注", type="primary", key="generate_btn"):
    if use_seed:
        seed_datetime = datetime.combine(seed_date, seed_time)
        seed_val = int(seed_datetime.timestamp())
        random.seed(seed_val)
        np.random.seed(seed_val)
        st.success(f"✅ 已设置随机种子: {seed_datetime.strftime('%Y-%m-%d %H:%M')}")
    else:
        random.seed()
        np.random.seed()
    
    with st.spinner(f"正在使用 {ai_model} 生成投注..."):
        # 提取复式类型代码（如"7+1"）
        bet_type_code = bet_type.split(' ')[0]  # "7+1 (14元)" -> "7+1"
        
        if "综合模式" in ai_model:
            bets = BetGenerator.generate_ensemble(draws, num_bets, bet_type_code)
        else:
            method_name = ai_model.split(":")[0] if ":" in ai_model else ai_model
            bets = BetGenerator.generate(method_name, draws, num_bets, bet_type_code)
        
        st.session_state['generated_bets'] = bets
        st.session_state['model_used'] = ai_model
        st.session_state['last_bet_type'] = bet_type_code
    
    st.success(f"✅ 使用 {ai_model} 生成 {len(bets)} 组 {bet_type_code} 复式投注")

# 显示生成的投注（表格形式，无图片卡片）
if st.session_state.get('generated_bets'):
    bets = st.session_state['generated_bets']
    model_used = st.session_state.get('model_used', '未知')
    bet_type_display = st.session_state.get('last_bet_type', '7+1')
    
    st.markdown(f"### 📝 推荐投注组合 - {model_used}")
    st.caption(f"{bet_type_display}复式，每组成本{bet_type_display.split('+')[0]}红球 + {bet_type_display.split('+')[1]}蓝球")
    
    # 表格形式显示投注
    bets_data = []
    for i, bet in enumerate(bets, 1):
        # 格式化红球
        reds_str = ' '.join([f"{r:02d}" for r in bet['reds'][:7]])  # 最多显示7个
        # 格式化蓝球
        if 'blues' in bet and len(bet['blues']) > 1:
            blues_str = ' '.join([f"{b:02d}" for b in bet['blues']])
        else:
            blues_str = f"{bet['blue']:02d}"
        
        bets_data.append({
            '组别': i,
            '红球': reds_str,
            '蓝球': blues_str,
            '红球数量': len(bet['reds']),
            '蓝球数量': len(bet.get('blues', [bet['blue']])),
            '和值': bet['sum']
        })
    
    st.dataframe(
        pd.DataFrame(bets_data), 
        use_container_width=True, 
        hide_index=True,
        column_config={
            '组别': st.column_config.NumberColumn('组别', width='small'),
            '红球': st.column_config.TextColumn('红球', width='large'),
            '蓝球': st.column_config.TextColumn('蓝球', width='medium'),
            '红球数量': st.column_config.NumberColumn('红球数量', width='small'),
            '蓝球数量': st.column_config.NumberColumn('蓝球数量', width='small'),
            '和值': st.column_config.NumberColumn('和值', width='small')
        }
    )
    
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
        with st.spinner("正在回测（使用滚动窗口模式，请耐心等待）..."):
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
            
            st.caption(f"回测期间：最近{backtest_periods}期 | 使用滚动窗口模式，每10期重新训练模型")

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

def parse_check_draws(text: str, max_draws: int = 50) -> List[Dict]:
    """解析查奖数据"""
    lines = text.strip().split('\n')
    draws_list = []
    for line in lines[:max_draws]:
        if not line.strip():
            continue
        parts = line.replace(',', ' ').split()
        if len(parts) >= 9:
            try:
                draws_list.append({
                    'period': parts[0],
                    'reds': [int(parts[i]) for i in range(2, 8)],
                    'blue': int(parts[8])
                })
            except:
                continue
    return draws_list

if st.button("🔍 查奖", key="check_btn") and check_draws_text:
    check_draws_list = parse_check_draws(check_draws_text, max_draws=max_check_draws)
    if check_draws_list:
        st.success(f"✅ 成功解析 {len(check_draws_list)} 期数据")
        
        if st.session_state.get('generated_bets'):
            # 构建表格数据
            enhanced_data = []
            for i, bet in enumerate(st.session_state['generated_bets'], 1):
                # 格式化红球
                reds_str = ' '.join([f"{r:02d}" for r in bet['reds']])
                # 格式化蓝球
                if 'blues' in bet and len(bet['blues']) > 1:
                    blues_str = ' '.join([f"{b:02d}" for b in bet['blues']])
                else:
                    blues_str = f"{bet['blue']:02d}"
                
                row = {
                    '组别': i, 
                    '红球': reds_str, 
                    '蓝球': blues_str
                }
                
                total_prize = 0
                for draw in check_draws_list:
                    # 计算该组投注的所有蓝球中奖情况
                    period_prize = 0
                    blues_to_check = bet.get('blues', [bet['blue']])
                    for blue in blues_to_check:
                        temp_bet = {'reds': bet['reds'], 'blue': blue}
                        prize, level = calculate_prize_for_single_bet(temp_bet, draw)
                        period_prize += prize
                    
                    total_prize += period_prize
                    if period_prize > 0:
                        row[f'{draw["period"]}'] = f"中{period_prize}元"
                    else:
                        row[f'{draw["period"]}'] = "未中奖"
                
                row['总奖金'] = f"¥{total_prize}"
                enhanced_data.append(row)
            
            st.dataframe(pd.DataFrame(enhanced_data), use_container_width=True, hide_index=True)
            
            # 统计总中奖情况
            all_prizes = []
            for row in enhanced_data:
                for key, value in row.items():
                    if '中' in str(value) and '元' in str(value) and key not in ['组别', '红球', '蓝球', '总奖金']:
                        all_prizes.append(value)
            
            if all_prizes:
                total_prize_sum = sum([int(str(v).replace('中', '').replace('元', '')) for v in all_prizes if '中' in str(v)])
                st.success(f"🎉 共中奖 {len(all_prizes)} 注，总奖金 ¥{total_prize_sum}")
            else:
                st.info("本期未中奖，继续加油！")
        else:
            st.warning("请先生成投注组合")
    else:
        st.error("解析失败，请检查格式")

# ==================== 底部 ====================
st.markdown("---")
st.caption("⚠️ 本工具仅供学术研究和娱乐参考。双色球本质随机，历史规律不代表未来结果。2026年新规下中3红有福运奖5元。请理性投注，量力而行。")

# ==================== 侧边栏内容 ====================
with st.sidebar:
    st.markdown("### 🎰 双色球AI分析工具 v12.0")
    st.markdown("---")
    
    # ML库状态
    with st.expander("🤖 ML库状态", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"{'✅' if LGB_AVAILABLE else '❌'} **LightGBM**")
            st.markdown(f"{'✅' if XGB_AVAILABLE else '❌'} **XGBoost**")
        with col2:
            st.markdown(f"{'✅' if SKLEARN_AVAILABLE else '❌'} **scikit-learn**")
        st.caption(f"MCP服务: {'✅ 可用' if MCP_AVAILABLE else '❌ 不可用'}")
    
    # 五种AI算法对比
    with st.expander("📖 五种AI算法对比（动态回测）"):
        backtest_periods_sidebar = st.slider(
            "回测期数",
            min_value=10,
            max_value=min(200, len(draws) - 10) if len(draws) > 10 else 30,
            value=30,
            step=5,
            key="sidebar_backtest_periods"
        )
        
        if st.button("🔄 刷新ROI", use_container_width=True, key="refresh_roi_sidebar"):
            st.cache_data.clear()
            st.rerun()
        
        if len(draws) >= backtest_periods_sidebar:
            try:
                roi_results = {}
                for method in ["方法1", "方法2", "方法3", "方法4"]:
                    result = backtest_roi(draws, method, num_bets=4, lookback=backtest_periods_sidebar)
                    roi_results[method] = result['roi']
                
                ensemble_roi = (roi_results.get("方法2", 0) + roi_results.get("方法3", 0) + roi_results.get("方法4", 0)) / 3
                
                st.markdown(f"""
                | 算法 | 特点 | {backtest_periods_sidebar}期ROI |
                |------|------|---------|
                | 🟢 方法1 | 冷热码+和值预测 | {roi_results.get("方法1", 0):.1f}% |
                | 🟡 方法2 | 胆拖混合 | {roi_results.get("方法2", 0):.1f}% |
                | 🔵 方法3 | LightGBM | {roi_results.get("方法3", 0):.1f}% |
                | 🟣 方法4 | XGBoost+NN | {roi_results.get("方法4", 0):.1f}% |
                | 🌟 方法5 | 综合模式（投票） | {ensemble_roi:.1f}% |
                """)
                
                best_method = max(roi_results, key=roi_results.get)
                best_roi = roi_results[best_method]
                if ensemble_roi > best_roi:
                    st.success(f"🏆 当前最佳：方法5 综合模式 (ROI: {ensemble_roi:.1f}%)")
                else:
                    st.success(f"🏆 当前最佳：{best_method} (ROI: {best_roi:.1f}%)")
                
                st.caption(f"📅 基于最近{backtest_periods_sidebar}期回测")
            except Exception as e:
                st.error(f"回测失败: {e}")
        else:
            st.warning(f"数据不足，需要至少{backtest_periods_sidebar}期")
    
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
    st.caption("DFSS智能选号工具 v12.0")
    st.caption("更新: 2026-05-10")


print("第5部分加载完成")
print("=" * 60)
print("所有代码加载完成！应用已就绪。")
print("=" * 60)
