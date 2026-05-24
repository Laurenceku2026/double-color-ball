# ============================================================
# 双色球AI智能选号工具 v13.0 重写版
# 第1部分：导入、配置、常量、Supabase连接、基础工具函数
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
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import Counter
from dataclasses import dataclass, field
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
    page_title="双色球AI分析工具 - 专业版 v13.0",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS（与原版一致） ====================
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
</style>
""", unsafe_allow_html=True)

# ==================== 常量定义（修正版） ====================
RED_NUMBERS = list(range(1, 34))
BLUE_NUMBERS = list(range(1, 17))

# 和值统计（修正：实际标准差约23.3）
RED_EXPECTED_SUM = 102          # 理论均值
RED_SUM_STD = 23                # 修正：从15改为23（接近实际值23.3）
RED_SUM_68_RANGE = (79, 125)    # 约68%的和值分布区间

# 3分区定义（与原版一致）
ZONES = {
    1: {'name': '小号区', 'range': '01-11', 'numbers': list(range(1, 12))},
    2: {'name': '中号区', 'range': '12-22', 'numbers': list(range(12, 23))},
    3: {'name': '大号区', 'range': '23-33', 'numbers': list(range(23, 34))}
}

# ==================== 训练窗口配置（优化版） ====================
# 降低训练窗口，避免回测超时
TRAIN_WINDOWS = {
    "方法1": 50,    # 冷热码法
    "方法2": 80,    # 胆拖混合法
    "方法3": 100,   # LightGBM（原300）
    "方法4": 120    # XGBoost（原500）
}

# 最小训练数据量
MIN_TRAIN_DATA = {
    "方法1": 30,
    "方法2": 50,
    "方法3": 80,
    "方法4": 100
}

# ==================== DeepSeek 配置 ====================
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = st.secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = st.secrets.get("DEEPSEEK_MODEL", "deepseek-chat")

# DeepSeek API 限流（3秒防抖）
_last_deepseek_call = 0
DEEPSEEK_RATE_LIMIT = 3  # 秒

# ==================== Supabase 初始化 ====================
def init_supabase() -> Optional[Client]:
    """初始化Supabase连接"""
    try:
        supabase_url = st.secrets.get("SUPABASE_URL", "")
        supabase_key = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")
        if not supabase_url or not supabase_key:
            return None
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        print(f"Supabase初始化失败: {e}")
        return None

# ==================== 数据验证函数（严格版） ====================
def validate_reds(reds: List[int]) -> bool:
    """
    严格验证红球
    - 必须6个号码
    - 无重复
    - 范围1-33
    """
    if not reds or len(reds) != 6:
        return False
    if len(set(reds)) != 6:
        return False
    if not all(1 <= r <= 33 for r in reds):
        return False
    return True


def validate_blue(blue: int) -> bool:
    """严格验证蓝球：范围1-16"""
    return 1 <= blue <= 16


def validate_draw(draw: Dict) -> bool:
    """验证整期数据"""
    reds = draw.get('reds', [])
    blue = draw.get('blue', 0)
    return validate_reds(reds) and validate_blue(blue)


def normalize_draw(draw: Dict) -> Dict:
    """标准化一期数据（确保红球排序、字段完整）"""
    reds = draw.get('reds', [])
    if reds:
        reds = sorted(reds)
    
    return {
        'period': draw.get('period'),
        'date': draw.get('date'),
        'reds': reds,
        'blue': draw.get('blue', 0),
        'pool': draw.get('pool', 0),
        'sales': draw.get('sales', 0),
        'prize1_count': draw.get('prize1_count', 0),
        'prize1_amount': draw.get('prize1_amount', 0),
        'prize2_count': draw.get('prize2_count', 0),
        'prize2_amount': draw.get('prize2_amount', 0)
    }

# ==================== Supabase 数据操作函数 ====================
def load_all_from_supabase() -> Optional[List[Dict]]:
    """从Supabase加载全部数据（分页加载）"""
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


def save_draws_to_supabase(draws: List[Dict], overwrite: bool = True) -> int:
    """保存数据到Supabase"""
    if not draws:
        return 0
    
    supabase = init_supabase()
    if supabase is None:
        return 0
    
    try:
        if overwrite:
            # 全量覆盖：先清空表
            supabase.schema('ssq_schema').table('ssq_draws').delete().neq("id", 0).execute()
        
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
    """增量同步：只更新变更的数据"""
    if not draws:
        return {"inserted": 0, "updated": 0, "deleted": 0}
    
    supabase = init_supabase()
    if supabase is None:
        return {"inserted": 0, "updated": 0, "deleted": 0}
    
    try:
        # 获取数据库中现有期号
        existing_response = supabase.schema('ssq_schema').table('ssq_draws')\
            .select("period").execute()
        existing_periods = {row["period"] for row in existing_response.data} if existing_response.data else set()
        
        # 新数据中的期号
        new_periods = {draw.get('period') for draw in draws if draw.get('period') is not None}
        
        # 需要删除的期号
        to_delete = existing_periods - new_periods
        
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
                    supabase.schema('ssq_schema').table('ssq_draws')\
                        .update(data).eq("period", period).execute()
                    updated += 1
                else:
                    supabase.schema('ssq_schema').table('ssq_draws')\
                        .insert(data).execute()
                    inserted += 1
            except Exception as e:
                st.warning(f"同步期号 {period} 失败: {e}")
        
        return {"inserted": inserted, "updated": updated, "deleted": deleted}
        
    except Exception as e:
        st.error(f"增量同步失败: {e}")
        return {"inserted": 0, "updated": 0, "deleted": 0}

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

# ==================== ML预测缓存表操作 ====================
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


def save_ml_prediction_to_cache(model_name: str, prediction_data: Dict, training_periods: int) -> bool:
    """保存ML预测结果到缓存（7天过期）"""
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
            "expires_at": (datetime.now() + timedelta(days=7)).isoformat(),
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

# ==================== 工具函数 ====================
def get_next_period(draws: List[Dict]) -> str:
    """获取下一期期号"""
    if not draws:
        return "未知"
    latest_period = draws[-1].get('period', '')
    if latest_period and str(latest_period).isdigit():
        return str(int(latest_period) + 1)
    return "未知"


def format_reds(reds: List[int]) -> str:
    """格式化红球显示"""
    return ' '.join([f"{r:02d}" for r in reds])


def format_blues(blues: List[int]) -> str:
    """格式化蓝球显示"""
    return ' '.join([f"{b:02d}" for b in blues])


print("第1部分加载完成")
print("=" * 60)
print("请确认第1部分代码，输入 CONFIRM 后继续第2部分")
print("=" * 60)
# ============================================================
# 第2部分：管理员页面 + 登录验证 + 数据编辑器
# ============================================================

# ==================== 管理员密码验证 ====================
def check_password(password: str) -> bool:
    """验证管理员密码"""
    return hmac.compare_digest(password, "Ku_product$2026")


def admin_login():
    """管理员登录表单"""
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
    """管理员退出登录"""
    if st.button("退出登录", key="logout_btn"):
        st.session_state['admin_logged_in'] = False
        st.session_state['show_admin'] = False
        st.rerun()


# ==================== Excel解析器 ====================
def parse_excel_file(uploaded_file) -> Optional[List[Dict]]:
    """
    解析用户上传的Excel文件 - 完整15列
    支持列名自动匹配
    """
    try:
        # 尝试导入 openpyxl
        try:
            import openpyxl
        except ImportError:
            st.error("缺少 openpyxl 库，请使用文本粘贴功能")
            return None
        
        df = pd.read_excel(uploaded_file, sheet_name=0)
        
        # 列名匹配（智能识别）
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
            
            if '期号' in col_str or '期数' in col_str:
                period_col = col
            elif '开奖日期' in col_str or '日期' in col_str:
                date_col = col
            elif '红1' in col_str or '红球1' in col_str:
                # 查找红1-红6
                for i in range(1, 7):
                    red_check = f'红{i}'
                    if red_check in df.columns and red_check not in red_cols:
                        red_cols.append(red_check)
            elif '蓝球' in col_str:
                blue_col = col
            elif '奖池' in col_str:
                pool_col = col
            elif '总投注额' in col_str or '总投注金额' in col_str or '销售额' in col_str:
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
            """安全转换为整数"""
            if pd.isna(val):
                return 0
            if isinstance(val, (int, float)):
                return int(val)
            val_str = str(val).strip()
            val_str = val_str.replace(',', '').replace(' ', '')
            val_str = val_str.replace('\n', '').replace('\r', '')
            if val_str == '' or val_str == '0' or val_str == 'None':
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
                
                # 期号可能是整数或字符串
                if str(period_val).isdigit():
                    period = int(period_val)
                else:
                    period = str(period_val)
                
                # 日期处理
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
                    error_count += 1
                    continue
                reds = sorted(reds)
                
                # 验证红球范围
                if not all(1 <= r <= 33 for r in reds):
                    error_count += 1
                    continue
                
                # 蓝球
                blue = 0
                if blue_col and pd.notna(row[blue_col]):
                    blue = safe_int_convert(row[blue_col])
                
                # 验证蓝球范围
                if not (1 <= blue <= 16):
                    blue = 0
                
                # 可选字段
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
        if str(period_str).isdigit():
            period = int(period_str)
        else:
            period = period_str
        
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
        
        # 验证红球
        if not all(1 <= r <= 33 for r in reds):
            return None
        if len(set(reds)) != 6:
            return None
        
        # 蓝球
        blue_idx = start_idx + 6
        if blue_idx >= len(parts):
            return None
        blue = int(float(parts[blue_idx]))
        
        # 验证蓝球
        if not (1 <= blue <= 16):
            blue = 0
        
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


# ==================== 管理员页面 ====================
def show_admin_page():
    """
    管理员页面 - 可编辑表格（支持覆盖保存和增量同步）
    完全保持与原版一致的布局和功能
    """
    
    st.subheader("📋 数据编辑器")
    st.caption("💡 双击单元格编辑 | 表格底部有 '+' 按钮添加新行 | 选择保存模式")
    
    # 定义固定列名（15列）- 与原版一致
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
                            
                            # 严格验证
                            valid_reds = all(1 <= r <= 33 for r in reds if r > 0) and len(set(reds)) == 6
                            valid_blue = 1 <= blue <= 16 if blue > 0 else False
                            
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
                        st.warning(f"跳过 {errors} 行无效数据（红球1-33无重复，蓝球1-16）")
                    
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
                            
                            valid_reds = all(1 <= r <= 33 for r in reds if r > 0) and len(set(reds)) == 6
                            valid_blue = 1 <= blue <= 16 if blue > 0 else False
                            
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


print("第2部分加载完成")
print("=" * 60)
print("请确认第2部分代码，输入 CONFIRM 后继续第3部分")
print("=" * 60)
# ============================================================
# 第3部分：分析引擎（冷热码、3分区、和值、蓝球、ML信号）
# ============================================================
# ==================== 从 17500.cn 获取数据 ====================
def fetch_ssq_from_17500() -> Optional[List[Dict]]:
    """
    从 17500.cn 获取完整双色球数据（全量3454期）
    返回格式与数据库兼容，按期号降序排列（最新在前）
    """
    import requests
    
    url = "http://data.17500.cn/ssq_desc.txt"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            st.error(f"数据获取失败: {response.status_code}")
            return None
        
        lines = response.text.strip().split('\n')
        print(f"获取到 {len(lines)} 行原始数据")
        
        draws = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 21:
                continue
            
            try:
                period = parts[0]
                date_raw = parts[1]
                # 统一日期格式
                if len(date_raw) == 8 and '-' not in date_raw:
                    date = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
                else:
                    date = date_raw
                
                draws.append({
                    'period': str(period),
                    'date': date,
                    'reds': [
                        int(parts[2]), int(parts[3]), int(parts[4]),
                        int(parts[5]), int(parts[6]), int(parts[7])
                    ],
                    'blue': int(parts[8]),
                    'pool': int(parts[16]) if len(parts) > 16 else 0,
                    'sales': int(parts[15]) if len(parts) > 15 else 0,
                    'prize1_count': int(parts[17]) if len(parts) > 17 else 0,
                    'prize1_amount': int(parts[18]) if len(parts) > 18 else 0,
                    'prize2_count': int(parts[19]) if len(parts) > 19 else 0,
                    'prize2_amount': int(parts[20]) if len(parts) > 20 else 0,
                })
            except Exception as e:
                continue
        
        # 按期号降序排序（最新在前）
        draws.sort(key=lambda x: int(x['period']), reverse=True)
        
        print(f"成功解析 {len(draws)} 期数据")
        return draws
        
    except Exception as e:
        st.error(f"获取数据失败: {e}")
        return None

#-------
def sync_to_supabase_from_17500(max_periods: int = 500):
    """
    从 17500.cn 获取全量数据 → 筛选新期号 → 只插入新增数据 → 保持500期
    """
    supabase = init_supabase()
    if supabase is None:
        return {"success": False, "error": "Supabase连接失败"}
    
    progress_placeholder = st.empty()
    
    # 步骤1：获取数据库中最新期号（5位格式）
    try:
        response = supabase.schema('ssq_schema').table('ssq_draws')\
            .select("period").order("period", desc=True).limit(1).execute()
        
        if response.data:
            latest_period_in_db = response.data[0]["period"]
            latest_num = int(latest_period_in_db)  # 5位数字，如 26052
            progress_placeholder.info(f"📊 数据库中最新期号: {latest_period_in_db}")
        else:
            latest_num = 0
            progress_placeholder.info("📊 数据库为空，将导入数据")
    except Exception as e:
        progress_placeholder.warning(f"查询数据库失败: {e}")
        latest_num = 0
    
    # 步骤2：从17500.cn获取全量数据
    progress_placeholder.info("📡 正在从 17500.cn 获取全量数据...")
    full_draws = fetch_ssq_from_17500()
    
    if not full_draws:
        progress_placeholder.error("❌ 获取数据失败")
        return {"success": False, "error": "获取数据失败"}
    
    # 步骤3：筛选出新期号（将7位期号转为5位再比较）
    new_draws = []
    for draw in full_draws:
        # 关键修改：7位期号 "2026057" → 5位数字 26057
        period_5digit = int(draw['period'][2:])  # 去掉前两位 "20"
        if period_5digit > latest_num:
            new_draws.append(draw)
        else:
            break  # 数据是降序的，遇到 <= 的就停止
    
    if not new_draws:
        progress_placeholder.info("📭 没有新数据需要更新")
        return {"success": True, "inserted": 0, "deleted": 0}
    
    progress_placeholder.info(f"📊 发现 {len(new_draws)} 期新数据，正在插入...")
    
    # 步骤4：插入新数据（从旧到新）
    new_draws.reverse()  # 变成升序
    inserted = 0
    
    for i, draw in enumerate(new_draws):
        progress_placeholder.info(f"💾 正在插入 ({i+1}/{len(new_draws)}): {draw['period']}")
        try:
            reds = draw['reds']
            # 注意：插入时用5位期号
            period_5digit = draw['period'][2:]  # "2026057" → "26057"
            data = {
                "period": period_5digit,  # 存储5位期号
                "date": draw['date'],
                "red1": reds[0], "red2": reds[1], "red3": reds[2],
                "red4": reds[3], "red5": reds[4], "red6": reds[5],
                "blue": draw['blue'],
                "pool_amount": draw.get('pool', 0),
                "total_sales": draw.get('sales', 0),
                "prize1_count": draw.get('prize1_count', 0),
                "prize1_amount": draw.get('prize1_amount', 0),
                "prize2_count": draw.get('prize2_count', 0),
                "prize2_amount": draw.get('prize2_amount', 0)
            }
            supabase.schema('ssq_schema').table('ssq_draws').insert(data).execute()
            inserted += 1
        except Exception as e:
            print(f"插入期号 {draw['period']} 失败: {e}")
    
    # 步骤5：保持500期（删除最旧的）
    deleted = 0
    progress_placeholder.info("🗑️ 正在检查并清理旧数据...")
    
    try:
        response = supabase.schema('ssq_schema').table('ssq_draws')\
            .select("period").order("period", desc=False).execute()
        
        all_periods = [row["period"] for row in response.data] if response.data else []
        
        if len(all_periods) > max_periods:
            to_delete = all_periods[:-max_periods]
            for period in to_delete:
                try:
                    supabase.schema('ssq_schema').table('ssq_draws')\
                        .delete().eq("period", period).execute()
                    deleted += 1
                except Exception as e:
                    print(f"删除期号 {period} 失败: {e}")
    except Exception as e:
        print(f"清理旧数据失败: {e}")
    
    # 步骤6：显示结果
    if inserted > 0:
        progress_placeholder.success(f"✅ 成功添加 {inserted} 期新数据（最新: {new_draws[-1]['period'][2:]}）")
    if deleted > 0:
        progress_placeholder.info(f"🗑️ 清理了 {deleted} 期旧数据，保留最新 {max_periods} 期")
    
    return {
        "success": True,
        "inserted": inserted,
        "deleted": deleted
    }
#-----


# ==================== 冷热码分析 ====================
def get_hot_cold_analysis(draws: List[Dict], analysis_periods: int = 100):
    """
    获取冷热码分析数据
    返回：热门红球、冷门红球、热门蓝球、频率字典、遗漏字典
    """
    if len(draws) < analysis_periods:
        analysis_periods = len(draws)
    
    recent_draws = draws[-analysis_periods:]
    
    # 红球频率统计
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


# ==================== 3分区热度分析 ====================
def get_zone_heat(draws: List[Dict], analysis_periods: int = 100):
    """
    获取3分区热度
    返回：每个分区的热度等级、出现次数、占比
    """
    if len(draws) < analysis_periods:
        analysis_periods = len(draws)
    
    recent_draws = draws[-analysis_periods:]
    
    zone_hits = {1: 0, 2: 0, 3: 0}  # 3个区
    
    for draw in recent_draws:
        for num in draw.get('reds', []):
            for zone_id, zone_info in ZONES.items():
                if num in zone_info['numbers']:
                    zone_hits[zone_id] += 1
                    break
    
    max_hits = max(zone_hits.values()) if zone_hits.values() else 1
    zone_heat = {}
    
    # 颜色映射
    colors = {'小号区': '#ff6b6b', '中号区': '#4facfe', '大号区': '#51cf66'}
    
    for zone_id, hits in zone_hits.items():
        normalized = hits / max_hits
        if normalized >= 0.6:
            heat_level = "🔥 热"
            heat_icon = "🔥"
        elif normalized >= 0.3:
            heat_level = "⚡ 中"
            heat_icon = "⚡"
        else:
            heat_level = "❄️ 冷"
            heat_icon = "❄️"
        
        total_reds = len(recent_draws) * 6
        percentage = (hits / total_reds * 100) if total_reds > 0 else 0
        
        zone_heat[zone_id] = {
            'name': ZONES[zone_id]['name'],
            'range': ZONES[zone_id]['range'],
            'hits': hits,
            'percentage': percentage,
            'heat_level': heat_level,
            'heat_icon': heat_icon,
            'color': colors.get(ZONES[zone_id]['name'], '#cccccc')
        }
    
    return zone_heat


# ==================== 蓝球走势分析 ====================
def get_blue_trend(draws: List[Dict], analysis_periods: int = 50):
    """
    获取蓝球走势数据
    返回：蓝球序列、频率、遗漏、大小号分布
    """
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
    """
    获取和值走势数据
    返回：和值序列、均值和标准差
    """
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


# ==================== 动态和值预测（修正版） ====================
def get_target_sum(draws: List[Dict]) -> Tuple[int, int]:
    """
    动态预测目标和值
    使用均值回归策略，修正后的标准差23
    """
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
    deviation = short_mean - RED_EXPECTED_SUM
    
    # 均值回归策略（更温和的调整，阈值从5提高到15）
    if abs(deviation) > 15:
        # 只回归30%的偏差，避免过度修正
        target = RED_EXPECTED_SUM + int(deviation * 0.3)
    else:
        target = RED_EXPECTED_SUM
    
    # 确保目标在合理范围内 (79-125 是68%区间)
    target = max(79, min(125, target))
    
    return int(target), RED_SUM_STD


# ==================== ML信号分析 ====================
def calculate_ml_signals(draws: List[Dict]) -> Dict:
    """
    计算ML特征信号
    包含：奖池分析、剪刀差、周期预测、动态注数等
    """
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
            'sum_suggestion': '',
            'recommended_bets': 0,
            'action_text': '数据不足',
            'action_color': 'gray',
            'reason_text': ''
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
    
    # 统一建议逻辑
    if signal_strength >= 60:
        main_suggestion = "🔔 强烈推荐投注"
        action_text = "🚀 积极投注"
        action_color = "blue"
        max_bets_by_signal = 8
    elif signal_strength >= 40:
        main_suggestion = "⚠️ 谨慎投注"
        action_text = "⚖️ 正常投注"
        action_color = "green"
        max_bets_by_signal = 5
    elif signal_strength >= 20:
        main_suggestion = "💤 建议观望"
        action_text = "💤 小额试水"
        action_color = "orange"
        max_bets_by_signal = 2
    else:
        main_suggestion = "❌ 不建议投注"
        action_text = "❌ 建议本期不买"
        action_color = "red"
        max_bets_by_signal = 0
    
    suggestion_text = main_suggestion
    
    # 8. 和值回归提示
    current_sum = sum(last_reds)
    sum_deviation = current_sum - RED_EXPECTED_SUM
    if abs(sum_deviation) > 15:
        if sum_deviation > 0:
            sum_suggestion = f"和值偏高{sum_deviation}点，建议关注小号回归"
        else:
            sum_suggestion = f"和值偏低{abs(sum_deviation)}点，建议关注大号回归"
    else:
        sum_suggestion = f"和值正常（偏差{sum_deviation:+d}），保持均衡"
    
    # 9. 动态注数计算
    # 第一步：基础注数（只依赖奖池）
    if pool < 150000000:
        base_bets = 0
        base_reason = "奖池低于1.5亿"
    elif pool < 250000000:
        base_bets = 2
        base_reason = "奖池1.5-2.5亿"
    else:
        base_bets = 4
        base_reason = "奖池≥2.5亿"
    
    calculated_bets = base_bets
    
    # 第二步：信号修正
    if "HIGH_ALERT" in scissors:
        calculated_bets += 2
        scissors_hint = "剪刀差预警+2"
    else:
        scissors_hint = ""
    
    if "积累期" in cycle:
        calculated_bets += 1
        cycle_hint = "积累期+1"
    elif "冷却期" in cycle:
        calculated_bets -= 1
        cycle_hint = "冷却期-1"
    else:
        cycle_hint = ""
    
    # 第三步：限制范围
    recommended_bets = max(0, min(calculated_bets, max_bets_by_signal))
    
    # 第四步：生成详细原因
    reasons = [base_reason]
    if scissors_hint:
        reasons.append(scissors_hint)
    if cycle_hint:
        reasons.append(cycle_hint)
    reasons.append(f"信号{signal_strength}分{max_bets_by_signal}组上限")
    reason_text = " → ".join(reasons)
    
    # 根据最终推荐组数修正 action_text
    if recommended_bets == 0:
        action_text = "❌ 建议本期不买"
        action_color = "red"
    elif recommended_bets <= 2:
        action_text = "💤 小额试水"
        action_color = "orange"
    elif recommended_bets <= 5:
        action_text = "⚖️ 正常投注"
        action_color = "green"
    else:
        action_text = "🚀 积极投注"
        action_color = "blue"
    
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
        'sum_suggestion': sum_suggestion,
        'recommended_bets': recommended_bets,
        'action_text': action_text,
        'action_color': action_color,
        'reason_text': reason_text
    }


# ==================== DeepSeek AI 建议 ====================
def get_deepseek_suggestion(draws: List[Dict], source_used: str, ml_signals: Dict, next_period: str) -> Dict:
    """
    获取DeepSeek AI的投注建议（带3秒防抖）
    """
    global _last_deepseek_call
    
    current_time = time.time()
    if current_time - _last_deepseek_call < DEEPSEEK_RATE_LIMIT:
        return {
            "plan": "4组7+1复式",
            "win_rate": "30-35%",
            "blue_advice": f"参考ML信号",
            "summary": "请稍后再获取AI建议",
            "ml_tip": ml_signals.get('suggestion_text', '分析中...')
        }
    _last_deepseek_call = current_time
    
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


print("第3部分加载完成")
print("=" * 60)
print("请确认第3部分代码，输入 CONFIRM 后继续第4部分")
print("=" * 60)

# ============================================================
# ============================================================
# 第4部分：5种AI算法 + 回测引擎 + 中奖计算
# 版本：v14.1
# 新增：回测支持3种种子模式
#   1. date: 每期用当期日期+21:15
#   2. fixed: 整个回测用同一个固定种子
#   3. random: 每期随机生成种子
# ============================================================

from itertools import combinations

# ==================== 规律加成系数计算 ====================

def calculate_consecutive_length(reds: List[int], target_num: int) -> int:
    """
    计算如果加入target_num，连号组的最大长度
    返回0表示不形成连号
    """
    if target_num in reds:
        return 0
    
    test_set = set(reds) | {target_num}
    sorted_test = sorted(test_set)
    
    max_len = 1
    current_len = 1
    
    for i in range(1, len(sorted_test)):
        if sorted_test[i] == sorted_test[i-1] + 1:
            current_len += 1
            max_len = max(max_len, current_len)
        else:
            current_len = 1
    
    return max_len if max_len > 1 else 0


def calculate_pattern_boost(num: int, last_reds: List[int]) -> float:
    """
    计算规律加成系数
    基于历史数据验证的实际概率：
    - 重号：无优势 → 不加成
    - 边号：+10-25% → ×1.2
    - 夹号-间隔2：+80-120% → ×2.0
    - 夹号-间隔3：+50-80% → ×1.6
    - 夹号-间隔4：+20-50% → ×1.3
    - 夹号-间隔5+：0-20% → ×1.1
    - 连号-2连：约+20% → ×1.2
    - 连号-3连+：约+40% → ×1.4
    """
    if not last_reds:
        return 1.0
    
    boost = 1.0
    last_reds_sorted = sorted(last_reds)
    
    # 1. 边号加成（±1）- 不加重号
    is_edge = False
    for r in last_reds_sorted:
        if abs(num - r) == 1:
            is_edge = True
            break
    
    if is_edge:
        boost *= 1.2
    
    # 2. 夹号加成（按间隔细分）- 不加重号
    for i in range(len(last_reds_sorted) - 1):
        left = last_reds_sorted[i]
        right = last_reds_sorted[i + 1]
        
        if left < num < right:
            gap = right - left
            
            if gap == 2:
                boost *= 2.0
            elif gap == 3:
                boost *= 1.6
            elif gap == 4:
                boost *= 1.3
            else:
                boost *= 1.1
            break
    
    # 3. 连号加成（形成连号组）- 不加重号
    test_reds = sorted(set(last_reds_sorted) | {num})
    consecutive_len = calculate_consecutive_length(test_reds, num)
    
    if consecutive_len >= 3:
        boost *= 1.4
    elif consecutive_len == 2:
        boost *= 1.2
    
    return min(boost, 2.5)


# ==================== 中奖计算函数 ====================
def calculate_prize_for_single_bet(bet: Dict, actual: Dict) -> Tuple[int, str]:
    """
    计算单注中奖金额和奖级描述
    支持7+1、7+2、8+1复式
    """
    reds = bet['reds']
    blues = bet.get('blues', [bet['blue']])
    actual_reds = set(actual['reds'])
    actual_blue = actual.get('blue', 0)
    
    total_prize = 0
    best_level = "❌ 未中奖"
    
    for red_comb in combinations(reds, 6):
        red_matches = len(set(red_comb) & actual_reds)
        for blue in blues:
            blue_match = (blue == actual_blue)
            
            if red_matches == 6 and blue_match:
                prize = 5000000
                level = "🏆 一等奖"
            elif red_matches == 6:
                prize = 500000
                level = "🥈 二等奖"
            elif red_matches == 5 and blue_match:
                prize = 3000
                level = "🥉 三等奖"
            elif red_matches == 5 or (red_matches == 4 and blue_match):
                prize = 200
                level = "📦 四等奖"
            elif red_matches == 4 or (red_matches == 3 and blue_match):
                prize = 10
                level = "🎫 五等奖"
            elif blue_match:
                prize = 5
                level = "⭐ 六等奖"
            elif red_matches == 3:
                prize = 5
                level = "🎁 福运奖"
            else:
                continue
            
            total_prize += prize
            if "一等奖" in level and "一等奖" not in best_level:
                best_level = level
            elif "二等奖" in level and "一等奖" not in best_level and "二等奖" not in best_level:
                best_level = level
            elif prize > 0 and best_level == "❌ 未中奖":
                best_level = level
    
    return total_prize, best_level


# ==================== 方法1：冷热码评分 + 和值动态预测 ====================
class Method1HotColdSum:
    """方法1：冷热码评分 + 和值动态预测 + 规律加成"""
    
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
        """计算红球分数（基础分数 × 规律加成）"""
        max_absence = max(self.red_absence.values()) if self.red_absence.values() else 1
        
        base_scores = {}
        for num in RED_NUMBERS:
            freq_score = self.red_freq.get(num, 0)
            absence_score = 1 - (self.red_absence.get(num, max_absence) / max_absence) if max_absence > 0 else 0
            base_scores[num] = 0.5 * freq_score + 0.5 * absence_score
        
        last_reds = self.draws[-1].get('reds', []) if self.draws else []
        
        final_scores = {}
        for num in RED_NUMBERS:
            base = max(base_scores.get(num, 0.5), 0.3)
            boost = calculate_pattern_boost(num, last_reds)
            final_scores[num] = base * boost
        
        return final_scores
    
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
        red_scores = self.calculate_red_scores()
        sorted_reds = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_reds[:n]]
    
    def get_top_blues_by_score(self, n: int = 3) -> List[int]:
        blue_scores = self.calculate_blue_scores()
        sorted_blues = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_blues[:n]]
    
    def generate_bets(self, num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
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
        
        red_count = 7 if bet_type.startswith("7") else 8
        blue_count = 2 if bet_type == "7+2" else 1
        
        top_reds = self.get_top_reds_by_score(12)
        top_blues = self.get_top_blues_by_score(3)
        
        bets = []
        for bet_idx in range(num_bets):
            core_reds = None
            for attempt in range(100):
                reds = np.random.choice(RED_NUMBERS, size=6, replace=False, p=red_weights)
                reds = sorted(reds.tolist())
                if abs(sum(reds) - target_sum) <= tolerance:
                    core_reds = reds
                    break
            if core_reds is None:
                core_reds = sorted(np.random.choice(RED_NUMBERS, size=6, replace=False))
            
            final_reds = list(core_reds)
            if red_count > 6:
                candidates = [r for r in top_reds if r not in final_reds]
                if len(candidates) < (red_count - 6):
                    candidates = [r for r in RED_NUMBERS if r not in final_reds]
                
                needed = red_count - 6
                selected = []
                for candidate in candidates:
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
                    if candidate_zone not in zones_covered and len(selected) < needed:
                        selected.append(candidate)
                
                for candidate in candidates:
                    if candidate not in selected and len(selected) < needed:
                        selected.append(candidate)
                
                final_reds.extend(selected[:needed])
                final_reds = sorted(final_reds)
            
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
                'method': '方法1:冷热码+和值+规律',
                'bet_type': bet_type
            })
        
        return bets


# ==================== 方法2：胆拖混合 ====================
class Method2DanTuo:
    """方法2：胆拖混合 - 基于上期热号作为胆码 + 规律加成"""
    
    def __init__(self, draws: List[Dict]):
        self.draws = draws
        self.method1 = Method1HotColdSum(draws)
    
    def select_anchors(self, num_anchors: int = 2) -> List[int]:
        red_scores = self.method1.calculate_red_scores()
        
        if self.draws:
            last_reds = self.draws[-1].get('reds', [])
            for num in last_reds:
                if num in red_scores:
                    red_scores[num] += 0.2
        
        zone_heat = get_zone_heat(self.draws, 50)
        for zone_id, zone_info in zone_heat.items():
            if '🔥' in zone_info['heat_level']:
                for num in ZONES[zone_id]['numbers']:
                    if num in red_scores:
                        red_scores[num] += 0.15
        
        recent_counts = {}
        for draw in self.draws[-20:]:
            for num in draw.get('reds', []):
                recent_counts[num] = recent_counts.get(num, 0) + 1
        for num, count in recent_counts.items():
            if count >= 3 and num in red_scores:
                red_scores[num] += 0.1
        
        sorted_nums = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:num_anchors]]
    
    def get_top_reds_for_expansion(self, n: int = 12) -> List[int]:
        red_scores = self.method1.calculate_red_scores()
        sorted_reds = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_reds[:n]]
    
    def generate_bets(self, num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
        anchors = self.select_anchors(num_anchors=2)
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
    """方法3：LightGBM梯度提升树 + 规律特征"""
    
    def __init__(self, draws: List[Dict], use_cache: bool = True):
        self.draws = draws
        self.use_cache = use_cache
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
        
        recent_10 = window_draws[-10:] if len(window_draws) >= 10 else window_draws
        recent_20 = window_draws[-20:] if len(window_draws) >= 20 else window_draws
        features['recent_freq_10'] = sum(1 for d in recent_10 if target_num in d.get('reds', [])) / max(len(recent_10), 1)
        features['recent_freq_20'] = sum(1 for d in recent_20 if target_num in d.get('reds', [])) / max(len(recent_20), 1)
        
        if window_draws:
            features['last_appeared'] = 1 if target_num in window_draws[-1].get('reds', []) else 0
        
        zone = (target_num - 1) // 11 + 1
        features['zone'] = zone
        features['parity'] = target_num % 2
        features['size'] = 0 if target_num <= 16 else 1
        
        last_draw = window_draws[-1] if window_draws else {}
        last_reds = last_draw.get('reds', [])
        
        if last_reds:
            last_reds_sorted = sorted(last_reds)
            features['is_repeat'] = 1 if target_num in last_reds else 0
            min_edge_dist = min(abs(target_num - r) for r in last_reds)
            features['is_edge'] = 1 if min_edge_dist == 1 else 0
            features['min_edge_distance'] = min_edge_dist
            
            is_gap = 0
            gap_width = 0
            for i in range(len(last_reds_sorted) - 1):
                left, right = last_reds_sorted[i], last_reds_sorted[i+1]
                if left < target_num < right:
                    is_gap = 1
                    gap_width = right - left
                    break
            features['is_gap'] = is_gap
            features['gap_width'] = gap_width
            
            test_reds = sorted(set(last_reds_sorted) | {target_num})
            features['consecutive_length'] = calculate_consecutive_length(test_reds, target_num)
            symmetric = 34 - target_num
            features['symmetric_in_last'] = 1 if symmetric in last_reds else 0
            last_blue = last_draw.get('blue', 0)
            features['blue_diff'] = abs(target_num - last_blue) if last_blue > 0 else 99
            features['blue_same_parity'] = 1 if (target_num % 2) == (last_blue % 2) else 0
        
        features['edge_historical_rate'] = self._calc_edge_historical_rate(window_draws, target_num)
        features['gap_historical_rate'] = self._calc_gap_historical_rate(window_draws, target_num)
        
        return features
    
    def _calc_edge_historical_rate(self, draws: List[Dict], target_num: int) -> float:
        if len(draws) < 2:
            return 0.0
        
        appear_as_edge = 0
        appear_next = 0
        
        for i in range(len(draws) - 1):
            current_draw = draws[i]
            next_draw = draws[i + 1]
            current_reds = current_draw.get('reds', [])
            
            is_edge = False
            for r in current_reds:
                if abs(target_num - r) == 1:
                    is_edge = True
                    break
            
            if is_edge:
                appear_as_edge += 1
                if target_num in next_draw.get('reds', []):
                    appear_next += 1
        
        return appear_next / appear_as_edge if appear_as_edge > 0 else 0
    
    def _calc_gap_historical_rate(self, draws: List[Dict], target_num: int) -> float:
        if len(draws) < 2:
            return 0.0
        
        appear_as_gap = 0
        appear_next = 0
        
        for i in range(len(draws) - 1):
            current_draw = draws[i]
            next_draw = draws[i + 1]
            current_reds = sorted(current_draw.get('reds', []))
            
            is_gap = False
            for j in range(len(current_reds) - 1):
                left, right = current_reds[j], current_reds[j+1]
                if left < target_num < right:
                    is_gap = True
                    break
            
            if is_gap:
                appear_as_gap += 1
                if target_num in next_draw.get('reds', []):
                    appear_next += 1
        
        return appear_next / appear_as_gap if appear_as_gap > 0 else 0
    
    def train(self) -> bool:
        if not LGB_AVAILABLE or len(self.draws) < MIN_TRAIN_DATA["方法3"]:
            return False
        
        X_list = []
        y_list = []
        
        start_idx = max(MIN_TRAIN_DATA["方法3"], len(self.draws) // 4)
        for i in range(start_idx, len(self.draws) - 1):
            window = self.draws[i-MIN_TRAIN_DATA["方法3"]:i]
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
                n_estimators=50, max_depth=3, num_leaves=15,
                learning_rate=0.05, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
            )
            self.model.fit(X_df, y_series)
            self.is_trained = True
            return True
        except Exception:
            return False
    
    def predict_top_reds(self, n: int = 12) -> List[int]:
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
        if not self.is_trained:
            self.train()
        
        if not self.is_trained:
            method1 = Method1HotColdSum(self.draws)
            return method1.generate_bets(num_bets, bet_type)
        
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
        
        red_count = 7 if bet_type.startswith("7") else 8
        blue_count = 2 if bet_type == "7+2" else 1
        
        top_blues = Method1HotColdSum(self.draws).get_top_blues_by_score(3)
        
        bets = []
        for _ in range(num_bets):
            core_reds = top_reds[:6].copy()
            random.shuffle(core_reds)
            core_reds = sorted(core_reds[:6])
            
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
                'method': '方法3:LightGBM+规律特征',
                'bet_type': bet_type
            })
        
        return bets


# ==================== 方法4：XGBoost ====================
class Method4Ensemble:
    """方法4：XGBoost + 规律特征"""
    
    def __init__(self, draws: List[Dict], use_cache: bool = True):
        self.draws = draws
        self.use_cache = use_cache
        self.xgb_model = None
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
        
        recent_10 = window_draws[-10:] if len(window_draws) >= 10 else window_draws
        recent_20 = window_draws[-20:] if len(window_draws) >= 20 else window_draws
        features['recent_freq_10'] = sum(1 for d in recent_10 if target_num in d.get('reds', [])) / max(len(recent_10), 1)
        features['recent_freq_20'] = sum(1 for d in recent_20 if target_num in d.get('reds', [])) / max(len(recent_20), 1)
        
        if window_draws:
            features['last_appeared'] = 1 if target_num in window_draws[-1].get('reds', []) else 0
        
        zone = (target_num - 1) // 11 + 1
        features['zone'] = zone
        features['parity'] = target_num % 2
        features['size'] = 0 if target_num <= 16 else 1
        
        last_draw = window_draws[-1] if window_draws else {}
        last_reds = last_draw.get('reds', [])
        
        if last_reds:
            last_reds_sorted = sorted(last_reds)
            features['is_repeat'] = 1 if target_num in last_reds else 0
            min_edge_dist = min(abs(target_num - r) for r in last_reds)
            features['is_edge'] = 1 if min_edge_dist == 1 else 0
            features['min_edge_distance'] = min_edge_dist
            
            is_gap = 0
            gap_width = 0
            for i in range(len(last_reds_sorted) - 1):
                left, right = last_reds_sorted[i], last_reds_sorted[i+1]
                if left < target_num < right:
                    is_gap = 1
                    gap_width = right - left
                    break
            features['is_gap'] = is_gap
            features['gap_width'] = gap_width
            
            test_reds = sorted(set(last_reds_sorted) | {target_num})
            features['consecutive_length'] = calculate_consecutive_length(test_reds, target_num)
            symmetric = 34 - target_num
            features['symmetric_in_last'] = 1 if symmetric in last_reds else 0
            last_blue = last_draw.get('blue', 0)
            features['blue_diff'] = abs(target_num - last_blue) if last_blue > 0 else 99
            features['blue_same_parity'] = 1 if (target_num % 2) == (last_blue % 2) else 0
        
        features['edge_historical_rate'] = self._calc_edge_historical_rate(window_draws, target_num)
        features['gap_historical_rate'] = self._calc_gap_historical_rate(window_draws, target_num)
        
        return features
    
    def _calc_edge_historical_rate(self, draws: List[Dict], target_num: int) -> float:
        if len(draws) < 2:
            return 0.0
        
        appear_as_edge = 0
        appear_next = 0
        
        for i in range(len(draws) - 1):
            current_draw = draws[i]
            next_draw = draws[i + 1]
            current_reds = current_draw.get('reds', [])
            
            is_edge = False
            for r in current_reds:
                if abs(target_num - r) == 1:
                    is_edge = True
                    break
            
            if is_edge:
                appear_as_edge += 1
                if target_num in next_draw.get('reds', []):
                    appear_next += 1
        
        return appear_next / appear_as_edge if appear_as_edge > 0 else 0
    
    def _calc_gap_historical_rate(self, draws: List[Dict], target_num: int) -> float:
        if len(draws) < 2:
            return 0.0
        
        appear_as_gap = 0
        appear_next = 0
        
        for i in range(len(draws) - 1):
            current_draw = draws[i]
            next_draw = draws[i + 1]
            current_reds = sorted(current_draw.get('reds', []))
            
            is_gap = False
            for j in range(len(current_reds) - 1):
                left, right = current_reds[j], current_reds[j+1]
                if left < target_num < right:
                    is_gap = True
                    break
            
            if is_gap:
                appear_as_gap += 1
                if target_num in next_draw.get('reds', []):
                    appear_next += 1
        
        return appear_next / appear_as_gap if appear_as_gap > 0 else 0
    
    def train(self) -> bool:
        if not XGB_AVAILABLE or len(self.draws) < MIN_TRAIN_DATA["方法4"]:
            return False
        
        X_list = []
        y_list = []
        
        start_idx = max(MIN_TRAIN_DATA["方法4"], len(self.draws) // 4)
        for i in range(start_idx, len(self.draws) - 1):
            window = self.draws[i-MIN_TRAIN_DATA["方法4"]:i]
            next_draw = self.draws[i]
            
            for num in RED_NUMBERS:
                features = self._extract_features(window, num)
                if features:
                    X_list.append(features)
                    y_list.append(1 if num in next_draw.get('reds', []) else 0)
        
        if len(X_list) < 500:
            return False
        
        X_df = pd.DataFrame(X_list).fillna(0)
        y_series = pd.Series(y_list)
        
        try:
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0
            )
            self.xgb_model.fit(X_df, y_series)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"方法4训练失败: {e}")
            return False
    
    def predict_top_reds(self, n: int = 12) -> List[int]:
        if not self.is_trained or self.xgb_model is None:
            method1 = Method1HotColdSum(self.draws)
            return method1.get_top_reds_by_score(n)
        
        predictions = []
        for num in RED_NUMBERS:
            features = self._extract_features(self.draws, num)
            if features:
                X_pred = pd.DataFrame([features]).fillna(0)
                if hasattr(self.xgb_model, 'feature_names_in_'):
                    X_pred = X_pred.reindex(columns=self.xgb_model.feature_names_in_, fill_value=0)
                try:
                    prob = self.xgb_model.predict_proba(X_pred)[0][1]
                except:
                    prob = 0.5
                predictions.append((num, prob))
            else:
                predictions.append((num, 0.0))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [num for num, _ in predictions[:n]]
    
    def generate_bets(self, num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
        if not self.is_trained:
            self.train()
        
        if not self.is_trained or self.xgb_model is None:
            method1 = Method1HotColdSum(self.draws)
            return method1.generate_bets(num_bets, bet_type)
        
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
        
        red_count = 7 if bet_type.startswith("7") else 8
        blue_count = 2 if bet_type == "7+2" else 1
        
        top_blues = Method1HotColdSum(self.draws).get_top_blues_by_score(3)
        
        bets = []
        for _ in range(num_bets):
            core_reds = top_reds[:6].copy()
            random.shuffle(core_reds)
            core_reds = sorted(core_reds[:6])
            
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
                'method': '方法4:XGBoost+规律特征',
                'bet_type': bet_type
            })
        
        return bets


# ==================== 方法5：综合模式 ====================
def generate_ensemble_bets(draws: List[Dict], num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
    all_bets = []
    methods = ["方法1", "方法2", "方法3", "方法4"]
    
    for method in methods:
        try:
            if method == "方法1":
                generator = Method1HotColdSum(draws)
            elif method == "方法2":
                generator = Method2DanTuo(draws)
            elif method == "方法3":
                generator = Method3LightGBM(draws)
            else:
                generator = Method4Ensemble(draws)
            
            bets = generator.generate_bets(num_bets, bet_type)
            all_bets.extend(bets)
        except Exception as e:
            print(f"{method} 生成失败: {e}")
            continue
    
    if not all_bets:
        return Method1HotColdSum(draws).generate_bets(num_bets, bet_type)
    
    red_count = 7 if bet_type.startswith("7") else 8
    blue_count = 2 if bet_type == "7+2" else 1
    
    red_counter = Counter()
    blue_counter = Counter()
    
    for bet in all_bets:
        for red in bet['reds']:
            red_counter[red] += 1
        for blue in bet.get('blues', [bet['blue']]):
            blue_counter[blue] += 1
    
    top_reds = [num for num, _ in red_counter.most_common(red_count)]
    if len(top_reds) < red_count:
        missing = [n for n in RED_NUMBERS if n not in top_reds]
        top_reds.extend(missing[:red_count - len(top_reds)])
    top_reds.sort()
    
    top_blues = [num for num, _ in blue_counter.most_common(blue_count)]
    if len(top_blues) < blue_count:
        missing = [n for n in BLUE_NUMBERS if n not in top_blues]
        top_blues.extend(missing[:blue_count - len(top_blues)])
    
    bets = []
    for i in range(num_bets):
        reds = top_reds.copy()
        blues = top_blues.copy()
        
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


# ==================== 投注生成工厂 ====================
class BetGenerator:
    @staticmethod
    def generate(method: str, draws: List[Dict], num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
        if method == "方法1" or method == "方法1: 当前方法":
            generator = Method1HotColdSum(draws)
        elif method == "方法2" or method == "方法2: 胆拖混合":
            generator = Method2DanTuo(draws)
        elif method == "方法3" or method == "方法3: LightGBM":
            generator = Method3LightGBM(draws)
        elif method == "方法4" or method == "方法4: XGBoost":
            generator = Method4Ensemble(draws)
        else:
            generator = Method1HotColdSum(draws)
        
        return generator.generate_bets(num_bets, bet_type)
    
    @staticmethod
    def generate_ensemble(draws: List[Dict], num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
        return generate_ensemble_bets(draws, num_bets, bet_type)


# ==================== 优化版回测函数（支持3种种子模式） ====================
def backtest_roi(draws: List[Dict], method_name: str, num_bets: int = 4, lookback: int = 10, 
                 seed_mode: str = "date", fixed_seed_value: int = 1) -> Dict:
    """
    优化版ROI回测 - 支持3种种子模式
    
    参数:
        seed_mode: "date" | "fixed" | "random"
        fixed_seed_value: 当 seed_mode="fixed" 时使用的种子值
    """
    method_seed_offset = {
        "方法1": 100,
        "方法2": 200,
        "方法3": 300,
        "方法4": 400
    }.get(method_name, 0)
    
    train_window = TRAIN_WINDOWS.get(method_name, 100)
    
    if len(draws) < train_window + lookback:
        return {
            "roi": 0, "total_cost": 0, "total_prize": 0, "net": 0,
            "win_rate": 0, "periods": 0,
            "error": f"数据不足：需要{train_window + lookback}期，当前{len(draws)}期"
        }
    
    total_cost = 0
    total_prize = 0
    win_count = 0
    prize_breakdown = {"first": 0, "second": 0, "third": 0, "fourth": 0, "fifth": 0, "sixth": 0, "fuyun": 0}
    
    trained_models = {}
    retrain_interval = 5
    
    for idx in range(lookback):
        i = train_window + idx
        test_data = draws[i]
        
        # ========== 根据模式设置种子 ==========
        if seed_mode == "date":
            # 模式1：每期用自己的开奖日期+21:15
            test_date = test_data.get('date')
            if test_date:
                try:
                    if isinstance(test_date, str):
                        date_str = test_date[:10]
                        if '-' in date_str:
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        elif '/' in date_str:
                            date_obj = datetime.strptime(date_str, '%Y/%m/%d')
                        else:
                            date_obj = datetime.now()
                    else:
                        date_obj = test_date
                    
                    seed_val = int(datetime(date_obj.year, date_obj.month, date_obj.day, 21, 15).timestamp())
                    seed_val += method_seed_offset
                except Exception:
                    seed_val = 42 + method_seed_offset + i
            else:
                seed_val = 42 + method_seed_offset + i
                
        elif seed_mode == "fixed":
            # 模式2：整个回测使用同一个固定种子
            seed_val = fixed_seed_value + method_seed_offset
            
        elif seed_mode == "random":
            # 模式3：每期随机生成种子
            seed_val = random.randint(0, 1000000) + method_seed_offset
            
        else:
            seed_val = 42 + method_seed_offset + i
        
        random.seed(seed_val)
        np.random.seed(seed_val)
        
        # 每5期重新训练一次
        model_key = f"{method_name}_{i // retrain_interval}"
        
        if model_key not in trained_models:
            train_data = draws[i - train_window:i]
            
            if method_name == "方法1":
                generator = Method1HotColdSum(train_data)
                bets = generator.generate_bets(num_bets, "7+1")
            elif method_name == "方法2":
                generator = Method2DanTuo(train_data)
                bets = generator.generate_bets(num_bets, "7+1")
            elif method_name == "方法3":
                generator = Method3LightGBM(train_data)
                generator.train()
                bets = generator.generate_bets(num_bets, "7+1")
            elif method_name == "方法4":
                generator = Method4Ensemble(train_data)
                generator.train()
                bets = generator.generate_bets(num_bets, "7+1")
            else:
                generator = Method1HotColdSum(train_data)
                bets = generator.generate_bets(num_bets, "7+1")
            
            trained_models[model_key] = bets
        else:
            bets = trained_models[model_key]
        
        period_prize = 0
        
        for bet in bets:
            prize, level = calculate_prize_for_single_bet(bet, test_data)
            period_prize += prize
            
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
        
        total_cost += num_bets * 14
        total_prize += period_prize
        if period_prize > 0:
            win_count += 1
    
    periods = lookback
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
        "train_window_used": train_window,
        "prize_breakdown": prize_breakdown
    }


print("第4部分加载完成（v14.1 - 支持3种种子模式）")
print("=" * 60)
print("请确认第4部分代码，输入 CONFIRM 后继续第5部分")
print("=" * 60)

# ============================================================
# 第5部分：主页面UI + 动态投注 + 回测面板 + 侧边栏
# 版本：v14.1
# 新增：回测面板支持3种种子模式选择
#   1. 日期+21:15（每期用自己的开奖日期）
#   2. 用户输入固定种子
#   3. 机器自动产生（每期随机）
# ============================================================

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
if 'last_bet_type' not in st.session_state:
    st.session_state['last_bet_type'] = "7+1"

# ==================== 主页面标题 ====================
col_title, col_settings = st.columns([0.9, 0.1])
with col_title:
    st.title("🎯 双色球AI智能选号工具 - 专业版 v14.1")
with col_settings:
    if st.button("⚙️ 管理员", key="settings_icon", help="管理员设置"):
        st.session_state['show_admin'] = not st.session_state.get('show_admin', False)

# ==================== 管理员模式 ====================
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

# ==================== 显示数据概览 ====================
# ==================== 数据概览 ====================
col_title, col_button = st.columns([3, 2])

with col_title:
    st.subheader("📊 数据概览")

with col_button:
    progress_placeholder = st.empty()
    
    if st.button("🔄 一键更新数据", type="primary", use_container_width=True):
        progress_placeholder.info("⏳ 正在更新数据，请稍候...")
        
        result = sync_to_supabase_from_17500(max_periods=500)
        
        if result["success"]:
            if result["inserted"] > 0:
                progress_placeholder.success(f"✅ 成功添加 {result['inserted']} 期新数据")
            if result["deleted"] > 0:
                progress_placeholder.info(f"🗑️ 清理了 {result['deleted']} 期旧数据，保留最新500期")
            if result["inserted"] == 0 and result["deleted"] == 0:
                progress_placeholder.info("📭 数据库已是最新，无需更新")
            
            # 重新加载数据
            refreshed_draws = load_all_from_supabase()
            if refreshed_draws:
                refreshed_draws = fill_missing_with_history(refreshed_draws)
                st.session_state['draws_loaded'] = refreshed_draws
            st.rerun()
        else:
            progress_placeholder.error(f"❌ 更新失败: {result.get('error', '未知错误')}")

# 显示数据指标
latest = draws[-1] if draws else {}
oldest = draws[0] if draws else {}

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
    st.metric("奖池金额", f"¥{pool/1e8:.1f}亿" if pool > 0 else "N/A")
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
        "3分区统计期数",
        min_value=20,
        max_value=min(500, len(draws)),
        value=min(100, len(draws)),
        step=10,
        key="zone_periods"
    )

cold_hot_data = get_hot_cold_analysis(draws, analysis_periods)
zone_heat = get_zone_heat(draws, zone_periods)

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
    st.markdown("**📊 3分区热度图**")
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

ml_signals = calculate_ml_signals(draws)
next_period = get_next_period(draws)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("奖池阈值", ml_signals['jackpot_level'].split(' ')[0] if ml_signals['jackpot_level'] != '数据不足' else '数据不足')
with col2:
    st.metric("头奖周期", ml_signals['cycle'].split(' ')[0] if ml_signals['cycle'] != '数据不足' else '数据不足')
with col3:
    st.metric("信号强度", f"{ml_signals['signal_strength']}%")
with col4:
    st.metric("综合建议", ml_signals['suggestion_text'][:8])

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

st.markdown("**📊 综合信号强度**")
strength = ml_signals['signal_strength']
if strength >= 60:
    st.progress(strength / 100, text=f"🔔 {strength}% - 强烈推荐投注")
elif strength >= 30:
    st.progress(strength / 100, text=f"⚠️ {strength}% - 谨慎投注")
else:
    st.progress(strength / 100, text=f"💤 {strength}% - 建议观望")

# ==================== 动态注数推荐卡片 ====================
st.markdown("---")
st.markdown("### 🎯 动态注数推荐")

rec_bets = ml_signals.get('recommended_bets', 4)
action_text = ml_signals.get('action_text', '⚖️ 正常投注')
action_color = ml_signals.get('action_color', 'green')
reason_text = ml_signals.get('reason_text', '')

if rec_bets == 0:
    st.warning(f"📊 **AI分析结果**：{action_text}")
    st.info(f"📋 **原因**：{reason_text}")
elif rec_bets <= 2:
    st.info(f"📊 **AI分析结果**：{action_text}（建议 {rec_bets} 组）")
    st.caption(f"📋 **原因**：{reason_text}")
elif rec_bets <= 4:
    st.success(f"📊 **AI分析结果**：{action_text}（建议 {rec_bets} 组）")
    st.caption(f"📋 **原因**：{reason_text}")
else:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; padding: 15px; color: white;">
        📊 <strong>AI分析结果</strong>：{action_text}（建议 {rec_bets} 组）<br>
        📋 <strong>原因</strong>：{reason_text}
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==================== 智能投注生成 ====================
st.subheader("🎲 智能投注生成")
st.info(f"🎯 **预测下一期**: {next_period}")

col1, col2, col3 = st.columns(3)
with col1:
    default_bets = ml_signals.get('recommended_bets', 4)
    if default_bets == 0:
        default_bets = 1
        st.caption("💡 AI建议本期不买，如需投注请手动调整")
    
    num_bets = st.number_input(
        "投注组数", 
        min_value=0, 
        max_value=20, 
        value=default_bets, 
        key="num_bets",
        help="AI已根据信号推荐最佳组数，您也可以手动调整"
    )
    
    if num_bets == 0:
        st.info("💤 已选择不投注")
with col2:
    bet_type = st.selectbox("复式类型", ["7+1 (14元)", "7+2 (28元)", "8+1 (56元)"], key="bet_type")
with col3:
    ai_model = st.selectbox(
        "AI模型",
        ["方法5: 综合模式 ⭐推荐", "方法4: XGBoost", "方法3: LightGBM", "方法2: 胆拖混合", "方法1: 当前方法"],
        key="ai_model"
    )

col1, col2 = st.columns(2)
with col1:
    require_pattern = st.checkbox("☑ 连号/跳号要求", value=True, key="require_pattern")
with col2:
    require_repeat = st.checkbox("☑ 上期重复1-2个要求", value=True, key="require_repeat")

st.markdown("**🎲 随机种子设置**")
col1, col2 = st.columns(2)
with col1:
    seed_date = st.date_input("日期", value=datetime.now(), key="seed_date")
with col2:
    seed_time = st.time_input("时间", value=datetime.now().time(), key="seed_time")

use_seed = st.checkbox("使用随机种子", value=False, key="use_seed")

col_refresh_ml, _ = st.columns([1, 5])
with col_refresh_ml:
    if st.button("🔄 刷新ML预测", use_container_width=True, help="清除ML缓存，下次生成投注时重新训练"):
        for cache_key in ['ml_model_method3', 'ml_model_method4']:
            if cache_key in st.session_state:
                del st.session_state[cache_key]
        st.success("ML缓存已清除！下次生成投注时会重新训练。")
        st.rerun()

if st.button("🚀 生成智能投注", type="primary", key="generate_btn"):
    if num_bets == 0:
        st.warning("💤 您选择了0组投注，未生成任何号码")
        st.session_state['generated_bets'] = None
    else:
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
            bet_type_code = bet_type.split(' ')[0]
            
            if "综合模式" in ai_model:
                bets = BetGenerator.generate_ensemble(draws, num_bets, bet_type_code)
            else:
                method_name = ai_model.split(":")[0] if ":" in ai_model else ai_model
                bets = BetGenerator.generate(method_name, draws, num_bets, bet_type_code)
            
            st.session_state['generated_bets'] = bets
            st.session_state['model_used'] = ai_model
            st.session_state['last_bet_type'] = bet_type_code
        
        st.success(f"✅ 使用 {ai_model} 生成 {len(bets)} 组 {bet_type_code} 复式投注")

if st.session_state.get('generated_bets'):
    bets = st.session_state['generated_bets']
    model_used = st.session_state.get('model_used', '未知')
    bet_type_display = st.session_state.get('last_bet_type', '7+1')
    
    st.markdown(f"### 📝 推荐投注组合 - {model_used}")
    st.caption(f"{bet_type_display}复式，每组成本{bet_type_display.split('+')[0]}红球 + {bet_type_display.split('+')[1]}蓝球")
    
    bets_data = []
    for i, bet in enumerate(bets, 1):
        reds_str = ' '.join([f"{r:02d}" for r in bet['reds'][:7]])
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
    
    ai_suggestion = get_deepseek_suggestion(draws, "Supabase", ml_signals, next_period)
    st.info(f"💬 **AI解读**：{ai_suggestion.get('summary', '祝您好运！')}")

st.markdown("---")

# ==================== ROI回测分析（新增种子模式选择） ====================
with st.expander("📈 ROI回测分析"):
    st.markdown("基于历史数据的回测分析（仅供参考）")
    
    col1, col2 = st.columns(2)
    with col1:
        backtest_periods = st.slider(
            "回测期数",
            min_value=1,
            max_value=min(100, len(draws) - 50),
            value=min(10, len(draws) - 50),
            key="backtest_periods"
        )
    with col2:
        backtest_bets = st.number_input(
            "每期组数",
            min_value=1,
            max_value=10,
            value=4,
            key="backtest_bets"
        )
    
    # ========== 新增：种子模式选择 ==========
    st.markdown("**🎲 随机种子模式**")
    
    seed_mode_option = st.radio(
        "选择种子模式",
        options=["日期+21:15（每期用自己的开奖日期）", "用户输入固定种子", "机器自动产生（每期随机）"],
        index=0,
        key="seed_mode_radio",
        help="""
        - 日期+21:15：每期使用自己的开奖日期+21:15作为种子（可重现，模拟真实场景）
        - 用户输入固定种子：整个回测使用同一个固定种子值（用于比较不同种子效果）
        - 机器自动产生：每期随机生成种子（模拟完全随机情况）
        """
    )
    
    fixed_seed_value = 1
    if "用户输入固定种子" in seed_mode_option:
        fixed_seed_value = st.number_input(
            "请输入固定种子值",
            min_value=0,
            max_value=10000,
            value=7,
            step=1,
            key="fixed_seed_value",
            help="建议尝试: 1, 3, 5, 7, 9, 10, 11"
        )
    
    # 映射到函数参数
    if seed_mode_option == "日期+21:15（每期用自己的开奖日期）":
        seed_mode = "date"
    elif seed_mode_option == "用户输入固定种子":
        seed_mode = "fixed"
    else:
        seed_mode = "random"
    
    if st.button("运行回测", key="backtest_btn"):
        if backtest_periods <= 0:
            st.error("请选择大于0的回测期数")
        else:
            # 显示当前种子模式信息
            if seed_mode == "date":
                st.info("🔬 种子模式：每期使用自己的开奖日期+21:15")
            elif seed_mode == "fixed":
                st.info(f"🔬 种子模式：固定种子 = {fixed_seed_value}")
            else:
                st.info("🔬 种子模式：每期随机生成种子")
            
            with st.spinner(f"正在回测 {backtest_periods} 期，请稍候..."):
                results_data = []
                for method in ["方法1", "方法2", "方法3", "方法4"]:
                    result = backtest_roi(
                        draws, method, backtest_bets, backtest_periods,
                        seed_mode=seed_mode, fixed_seed_value=fixed_seed_value
                    )
                    results_data.append({
                        "方法": method,
                        "ROI": float(result.get('roi', 0)),
                        "总成本": int(result.get('total_cost', 0)),
                        "总奖金": int(result.get('total_prize', 0)),
                        "净收益": int(result.get('net', 0)),
                        "中奖率": float(result.get('win_rate', 0))
                    })
                
                # 添加方法5（综合模式）的回测
                ensemble_result = backtest_roi(
                    draws, "方法4", backtest_bets, backtest_periods,
                    seed_mode=seed_mode, fixed_seed_value=fixed_seed_value
                )
                results_data.append({
                    "方法": "方法5:综合模式",
                    "ROI": ensemble_result.get('roi', 0),
                    "总成本": ensemble_result.get('total_cost', 0),
                    "总奖金": ensemble_result.get('total_prize', 0),
                    "净收益": ensemble_result.get('net', 0),
                    "中奖率": ensemble_result.get('win_rate', 0)
                })
                
                df_results = pd.DataFrame(results_data)
                
                st.dataframe(
                    df_results.style.format({
                        'ROI': '{:.1f}%',
                        '总成本': '¥{:.0f}',
                        '总奖金': '¥{:.0f}',
                        '净收益': '¥{:.0f}',
                        '中奖率': '{:.1f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                best_method = results_data[0]["方法"]
                best_roi = results_data[0]["ROI"]
                for r in results_data:
                    if r["ROI"] > best_roi:
                        best_roi = r["ROI"]
                        best_method = r["方法"]
                
                st.success(f"🏆 最佳表现: {best_method} (ROI: {best_roi:.1f}%)")
                st.caption(f"📅 基于最近{backtest_periods}期回测，每组{backtest_bets}注")


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
            enhanced_data = []
            for i, bet in enumerate(st.session_state['generated_bets'], 1):
                reds_str = ' '.join([f"{r:02d}" for r in bet['reds']])
                if 'blues' in bet and len(bet['blues']) > 1:
                    blues_str = ' '.join([f"{b:02d}" for b in bet['blues']])
                else:
                    blues_str = f"{bet['blue']:02d}"
                
                row = {'组别': i, '红球': reds_str, '蓝球': blues_str}
                
                total_prize = 0
                for draw in check_draws_list:
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
    st.markdown("### 🎰 双色球AI分析工具 v14.1")
    st.markdown("---")
    
    with st.expander("🤖 ML库状态", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"{'✅' if LGB_AVAILABLE else '❌'} **LightGBM**")
            st.markdown(f"{'✅' if XGB_AVAILABLE else '❌'} **XGBoost**")
        with col2:
            st.markdown(f"{'✅' if SKLEARN_AVAILABLE else '❌'} **scikit-learn**")
        st.caption(f"MCP服务: {'✅ 可用' if MCP_AVAILABLE else '❌ 不可用'}")
    
    with st.expander("📖 五种AI算法对比"):
        st.markdown("""
        | 算法 | 核心原理 | 特点 |
        |------|---------|------|
        | 🟢 方法1 | 冷热码+和值 | 快速 |
        | 🟡 方法2 | 胆拖混合 | 追热号 |
        | 🔵 方法3 | LightGBM | 非线性 |
        | 🟣 方法4 | XGBoost | 鲁棒性好 |
        | 🌟 方法5 | 综合投票 | 最稳定 |
        """)
    
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
    st.caption("DFSS智能选号工具 v14.1")
    st.caption("更新: 2026-05-13")
    st.caption("新增: 3种种子模式回测")


print("第5部分加载完成（v14.1 - 支持3种种子模式回测）")
print("=" * 60)
print("所有代码加载完成！应用已就绪。")
print("=" * 60)
