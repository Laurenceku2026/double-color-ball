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

# ==================== 训练窗口配置（修正版 v15.0） ====================

# 选号方法训练窗口（用于生成投注）
TRAIN_WINDOWS = {
    "方法1": 100,    # 新规则系统 - 需要100期分析疏密周期和蓝球规律
    "方法2": 100,    # 胆拖混合 - 依赖方法1，使用相同窗口
    "方法3": 100,    # LightGBM - ML模型
    "方法4": 120,    # XGBoost - ML模型需要更多数据
}

# 方法5综合模式使用最大窗口
METHOD5_WINDOW = 120

# ML智能分析引擎固定窗口（用于判断是否投注，与选号方法无关）
ML_SIGNAL_WINDOW = 50  # 回测验证的最佳窗口

# 最小训练数据量（对应调整）
MIN_TRAIN_DATA = {
    "方法1": 50,     # 新规则系统最小需要50期
    "方法2": 50,     # 胆拖混合最小50期
    "方法3": 80,     # LightGBM最小80期
    "方法4": 100,    # XGBoost最小100期
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

#---------------
def incremental_sync_draws(draws: List[Dict]) -> Dict:
    """增量同步：只更新变更的数据（优化版）"""
    if not draws:
        return {"inserted": 0, "updated": 0, "deleted": 0}
    
    supabase = init_supabase()
    if supabase is None:
        return {"inserted": 0, "updated": 0, "deleted": 0}
    
    try:
        # ========== 统一期号为字符串 ==========
        normalized_draws = []
        for draw in draws:
            if draw.get('period') is None:
                continue
            normalized_draw = draw.copy()
            normalized_draw['period'] = str(draw['period'])
            normalized_draws.append(normalized_draw)
        
        # 获取数据库中现有期号（统一转为字符串）
        existing_response = supabase.schema('ssq_schema').table('ssq_draws')\
            .select("period").execute()
        existing_periods = {str(row["period"]) for row in existing_response.data} if existing_response.data else set()
        
        # 新数据中的期号
        new_periods = {draw['period'] for draw in normalized_draws if draw.get('period') is not None}
        
        # 需要删除的期号
        to_delete = existing_periods - new_periods
        
        # 需要新增的期号
        to_insert = new_periods - existing_periods
        
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
        
        # 准备需要插入/更新的数据（只处理新增和已存在的）
        upsert_data = []
        for draw in normalized_draws:
            period = draw['period']
            reds = draw.get('reds', [])
            upsert_data.append({
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
            })
        
        # 使用 upsert 批量操作（一次请求处理所有新增和更新）
        if upsert_data:
            result = supabase.schema('ssq_schema').table('ssq_draws')\
                .upsert(upsert_data, on_conflict='period').execute()
            inserted = len(to_insert)
            updated = len(new_periods) - len(to_insert)
        
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
                
                # ========== 修改：期号统一转为字符串 ==========
                if '.' in str(period_val):
                    period = str(period_val).split('.')[0]
                else:
                    period = str(period_val).strip()
                
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
                    'period': period,  # 已经是字符串
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
        period_str = parts[0].strip()
        
        # ========== 修改：期号统一转为字符串 ==========
        # 去除可能的 .0 后缀（如 "26051.0" -> "26051"）
        if '.' in period_str:
            period_str = period_str.split('.')[0]
        period = str(period_str)
        
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
            'period': period,  # 现在是字符串
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
    
    # 配置列类型（期号强制为文本）
    column_config = {
        "期号": st.column_config.TextColumn("期号", required=True),
        "开奖日期": st.column_config.TextColumn("开奖日期"),
        "红1": st.column_config.NumberColumn("红1", min_value=1, max_value=33, step=1),
        "红2": st.column_config.NumberColumn("红2", min_value=1, max_value=33, step=1),
        "红3": st.column_config.NumberColumn("红3", min_value=1, max_value=33, step=1),
        "红4": st.column_config.NumberColumn("红4", min_value=1, max_value=33, step=1),
        "红5": st.column_config.NumberColumn("红5", min_value=1, max_value=33, step=1),
        "红6": st.column_config.NumberColumn("红6", min_value=1, max_value=33, step=1),
        "蓝球": st.column_config.NumberColumn("蓝球", min_value=1, max_value=16, step=1),
        "奖池奖金(元)": st.column_config.NumberColumn("奖池奖金(元)", step=1),
        "一等奖注数": st.column_config.NumberColumn("一等奖注数", step=1),
        "一等奖奖金(元)": st.column_config.NumberColumn("一等奖奖金(元)", step=1),
        "二等奖注数": st.column_config.NumberColumn("二等奖注数", step=1),
        "二等奖奖金(元)": st.column_config.NumberColumn("二等奖奖金(元)", step=1),
        "总投注额(元)": st.column_config.NumberColumn("总投注额(元)", step=1),
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
        
        # ========== 全量覆盖保存 ==========
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
                            
                            # ========== 期号统一转为字符串 ==========
                            period_str = str(row['期号']).strip()
                            if '.' in period_str:
                                period_str = period_str.split('.')[0]
                            period = period_str
                            
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
        
        # ========== 增量同步保存 ==========
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
                            
                            # ========== 期号统一转为字符串 ==========
                            period_str = str(row['期号']).strip()
                            if '.' in period_str:
                                period_str = period_str.split('.')[0]
                            period = period_str
                            
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
    st.markdown("---")

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
    从 17500.cn 获取全量数据 → 补齐到500期（新数据 + 缺失的旧数据）
    """
    supabase = init_supabase()
    if supabase is None:
        return {"success": False, "error": "Supabase连接失败"}
    
    progress_placeholder = st.empty()
    
    # ========== 步骤1：获取数据库当前状态 ==========
    try:
        # 获取最新期号（降序，取第一条）
        response_latest = supabase.schema('ssq_schema').table('ssq_draws')\
            .select("period").order("period", desc=True).limit(1).execute()
        
        # 获取最早期号（升序，取第一条）
        response_oldest = supabase.schema('ssq_schema').table('ssq_draws')\
            .select("period").order("period").limit(1).execute()
        
        # 获取总期数
        response_count = supabase.schema('ssq_schema').table('ssq_draws')\
            .select("period", count="exact").execute()
        
        if response_latest.data and response_oldest.data:
            latest_in_db = response_latest.data[0]["period"]
            oldest_in_db = response_oldest.data[0]["period"]
            current_count = response_count.count
            
            progress_placeholder.info(f"📊 数据库状态: {current_count}期 ({oldest_in_db} ~ {latest_in_db})")
        else:
            latest_in_db = None
            oldest_in_db = None
            current_count = 0
            progress_placeholder.info("📊 数据库为空，将初始化500期数据")
            
    except Exception as e:
        progress_placeholder.error(f"查询数据库失败: {e}")
        return {"success": False, "error": f"数据库查询失败: {e}"}
    
    # ========== 步骤2：从17500.cn获取全量数据 ==========
    progress_placeholder.info("📡 正在从 17500.cn 获取全量数据...")
    full_draws = fetch_ssq_from_17500()
    
    if not full_draws:
        progress_placeholder.error("❌ 获取数据失败")
        return {"success": False, "error": "获取数据失败"}
    
    # ========== 步骤3：确定需要补充的数据 ==========
    to_insert = []
    
    if current_count == 0:
        # 数据库为空：直接取最新的500期
        to_insert = full_draws[:max_periods]
        progress_placeholder.info(f"📊 数据库为空，将初始化最新 {len(to_insert)} 期数据")
    
    else:
        needed_count = max_periods - current_count
        
        # 3.1 补充新数据（比最新期号更大的）
        latest_5digit = int(latest_in_db)
        
        new_draws = []
        for draw in full_draws:
            period_5digit = int(draw['period'][2:])
            if period_5digit > latest_5digit:
                new_draws.append(draw)
            else:
                break
        
        # 3.2 如果还不够500期，补充旧数据（比最早期号更小的）
        old_draws = []
        if needed_count > len(new_draws):
            need_old = needed_count - len(new_draws)
            oldest_5digit = int(oldest_in_db)
            
            candidates = []
            for draw in full_draws:
                period_5digit = int(draw['period'][2:])
                if period_5digit < oldest_5digit:
                    candidates.append(draw)
            
            if candidates:
                old_draws = candidates[-need_old:] if len(candidates) > need_old else candidates
                old_draws.reverse()
        
        # 合并：从旧到新排列
        to_insert = old_draws + new_draws
        to_insert.reverse()
        
        progress_placeholder.info(f"📊 需要补充: {len(new_draws)}期新数据 + {len(old_draws)}期旧数据 = {len(to_insert)}期")
    
    # ========== 步骤4：插入数据 ==========
    if not to_insert:
        progress_placeholder.info("📭 数据库已是最新且满500期，无需更新")
        return {"success": True, "inserted": 0, "deleted": 0}
    
    progress_placeholder.info(f"💾 正在插入 {len(to_insert)} 期数据...")
    inserted = 0
    
    for i, draw in enumerate(to_insert):
        progress_placeholder.info(f"💾 插入进度: {i+1}/{len(to_insert)} (期号: {draw['period'][2:]})")
        try:
            reds = draw['reds']
            period_5digit = draw['period'][2:]
            data = {
                "period": period_5digit,
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
    
    # ========== 步骤5：保持500期 ==========
    deleted = 0
    progress_placeholder.info("🗑️ 正在检查并清理旧数据...")
    
    try:
        response = supabase.schema('ssq_schema').table('ssq_draws')\
            .select("period").order("period").execute()
        
        all_periods = [row["period"] for row in response.data] if response.data else []
        
        if len(all_periods) > max_periods:
            to_delete = all_periods[:len(all_periods) - max_periods]
            for period in to_delete:
                try:
                    supabase.schema('ssq_schema').table('ssq_draws')\
                        .delete().eq("period", period).execute()
                    deleted += 1
                except Exception as e:
                    print(f"删除期号 {period} 失败: {e}")
    except Exception as e:
        print(f"清理旧数据失败: {e}")
    
    # ========== 步骤6：显示结果 ==========
    if inserted > 0:
        progress_placeholder.success(f"✅ 成功添加 {inserted} 期数据")
    if deleted > 0:
        progress_placeholder.info(f"🗑️ 清理了 {deleted} 期旧数据，保持 {max_periods} 期")
    
    return {
        "success": True,
        "inserted": inserted,
        "deleted": deleted
    }
#-----
#----------
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

#----------------
# ==================== 正弦拟合和值预测（公共函数） ====================
def sine_fit_predict_sum(recent_sums: List[int]) -> int:
    """
    正弦拟合预测下一期和值（纯函数）
    
    参数:
        recent_sums: 最近10期和值列表
    返回:
        预测的和值（80-125范围内）
    """
    from scipy.optimize import curve_fit
    
    if len(recent_sums) < 10:
        return int(round(np.mean(recent_sums))) if recent_sums else 102
    
    def sine_func(x, A, omega, phi, C):
        return A * np.sin(omega * x + phi) + C
    
    x = np.arange(len(recent_sums))
    y = np.array(recent_sums)
    
    # 初始参数猜测
    A_guess = (np.max(y) - np.min(y)) / 2
    C_guess = np.mean(y)
    omega_guess = 2 * np.pi / 6.5  # 周期约6.5期
    
    try:
        params, _ = curve_fit(
            sine_func, x, y,
            p0=[A_guess, omega_guess, 0, C_guess],
            bounds=([0, 2*np.pi/15, -np.pi, 80], 
                    [50, 2*np.pi/4, np.pi, 125]),
            maxfev=2000
        )
        A, omega, phi, C = params
        next_val = sine_func(len(recent_sums), A, omega, phi, C)
        return max(80, min(125, int(round(next_val))))
    except:
        # 拟合失败时降级使用均值回归
        short_mean = np.mean(recent_sums)
        deviation = short_mean - 102
        if abs(deviation) > 10:
            target = 102 + int(deviation * 0.5)
        else:
            target = 102
        return max(80, min(125, target))


def get_target_sum_sine(draws: List[Dict], tolerance: int = 12) -> Tuple[int, int]:
    """
    获取正弦拟合预测的和值（供UI和选号使用）
    
    参数:
        draws: 历史开奖数据
        tolerance: 容差，默认12
    返回:
        (预测和值, 容差)
    """
    if len(draws) < 10:
        return 102, tolerance
    
    # 获取最近10期和值
    recent_sums = []
    for draw in draws[-10:]:
        reds = draw.get('reds', [])
        if reds:
            recent_sums.append(sum(reds))
    
    if len(recent_sums) < 10:
        return 102, tolerance
    
    target = sine_fit_predict_sum(recent_sums)
    return target, tolerance
#---------------------
# ==================== 7期移动平均和值预测（统一范围 ±12） ====================

def get_target_sum_range(draws: List[Dict], window: int = 7, tolerance: int = 12) -> Tuple[int, int]:
    """
    基于移动平均的红球和值范围预测
    
    参数:
        window: 移动平均窗口期（默认7期）
        tolerance: 容差（默认12）
    
    返回:
        (下限, 上限)
    """
    if len(draws) < window:
        return 80, 125  # 默认范围
    
    recent_sums = []
    for draw in draws[-window:]:
        reds = draw.get('reds', [])
        if reds:
            recent_sums.append(sum(reds))
    
    if not recent_sums:
        return 80, 125
    
    mean_val = np.mean(recent_sums)
    lower = max(50, int(mean_val - tolerance))
    upper = min(175, int(mean_val + tolerance))
    
    return lower, upper


def generate_target_sum_by_range(draws: List[Dict]) -> int:
    """在预测范围内随机生成一个和值"""
    lower, upper = get_target_sum_range(draws)
    return random.randint(lower, upper)


# ==================== 正弦拟合和值预测（统一返回范围 ±12） ====================

def get_target_sum_sine_range(draws: List[Dict], tolerance: int = 12) -> Tuple[int, int]:
    """
    正弦拟合预测和值范围
    
    参数:
        draws: 历史开奖数据
        tolerance: 容差（默认12）
    
    返回:
        (下限, 上限)
    """
    if len(draws) < 10:
        return 80, 125
    
    recent_sums = []
    for draw in draws[-10:]:
        reds = draw.get('reds', [])
        if reds:
            recent_sums.append(sum(reds))
    
    if len(recent_sums) < 10:
        return 80, 125
    
    target = sine_fit_predict_sum(recent_sums)
    lower = max(50, target - tolerance)
    upper = min(175, target + tolerance)
    
    return lower, upper


def generate_target_sum_by_sine(draws: List[Dict]) -> int:
    """在正弦拟合预测范围内随机生成一个和值"""
    lower, upper = get_target_sum_sine_range(draws)
    return random.randint(lower, upper)


# ==================== 7期移动平均蓝球预测（环形 ±3） ====================

def get_blue_range(draws: List[Dict], window: int = 7, radius: int = 3) -> List[int]:
    """
    基于移动平均的蓝球环形范围预测
    
    参数:
        window: 移动平均窗口期（默认7期）
        radius: 半径（默认3，共7个号码）
    
    返回:
        候选蓝球列表（环形，7个号码）
    """
    if len(draws) < window:
        return list(range(1, 17))  # 默认全部
    
    recent_blues = []
    for draw in draws[-window:]:
        blue = draw.get('blue', 0)
        if 1 <= blue <= 16:
            recent_blues.append(blue)
    
    if not recent_blues:
        return list(range(1, 17))
    
    center = int(round(np.mean(recent_blues)))
    
    # 环形生成 ±radius 范围内的号码
    candidates = []
    for offset in range(-radius, radius + 1):
        num = center + offset
        if num < 1:
            num = 16 + num
        elif num > 16:
            num = num - 16
        candidates.append(num)
    
    # 去重并保持顺序
    seen = set()
    unique_candidates = []
    for num in candidates:
        if num not in seen:
            seen.add(num)
            unique_candidates.append(num)
    
    return unique_candidates


def select_blue_by_range(draws: List[Dict]) -> int:
    """从预测范围中随机选择一个蓝球"""
    candidates = get_blue_range(draws)
    return random.choice(candidates)


# ==================== 正弦拟合蓝球预测（环形 ±3，共7个候选） ====================

def get_blue_sine_candidates(draws: List[Dict], window: int = 8, radius: int = 3) -> List[int]:
    """
    正弦拟合预测蓝球候选池（环形，±3）
    
    参数:
        draws: 历史开奖数据
        window: 正弦拟合窗口（默认8期）
        radius: 半径（默认3，共7个号码）
    
    返回:
        候选蓝球列表（环形，7个号码）
    """
    if len(draws) < window:
        return list(range(1, 17))
    
    blue_sequence = [d.get('blue', 0) for d in draws[-window:] if d.get('blue', 0) > 0]
    if len(blue_sequence) < 6:
        return list(range(1, 17))
    
    prediction = sine_fit_predict_blue(blue_sequence)
    
    # 环形生成 ±radius 范围内的号码
    candidates = []
    for offset in range(-radius, radius + 1):
        num = prediction + offset
        if num < 1:
            num = 16 + num
        elif num > 16:
            num = num - 16
        candidates.append(num)
    
    # 去重并保持顺序
    seen = set()
    unique_candidates = []
    for num in candidates:
        if num not in seen:
            seen.add(num)
            unique_candidates.append(num)
    
    return unique_candidates


def select_blue_by_sine(draws: List[Dict]) -> int:
    """从正弦拟合候选池中随机选择一个蓝球"""
    candidates = get_blue_sine_candidates(draws)
    return random.choice(candidates)


# ==================== 统一蓝球选择函数 ====================

def select_blue_by_method(draws: List[Dict], method: str) -> int:
    """
    根据指定的方法选择蓝球
    
    参数:
        draws: 历史开奖数据
        method: "正弦拟合" 或 "7期均值"
    
    返回:
        选中的蓝球号码
    """
    if method == "正弦拟合":
        candidates = get_blue_sine_candidates(draws)
    else:
        candidates = get_blue_range(draws)
    
    return random.choice(candidates)


# ==================== 蓝球正弦拟合预测（保留用于绘图） ====================

def sine_fit_predict_blue(blue_sequence: List[int]) -> int:
    """
    正弦拟合预测下一期蓝球（纯函数）
    
    参数:
        blue_sequence: 最近8期蓝球号码列表
    返回:
        预测的蓝球号码（1-16范围内）
    """
    from scipy.optimize import curve_fit
    
    if len(blue_sequence) < 6:
        return int(round(np.mean(blue_sequence))) if blue_sequence else 8
    
    def sine_func(x, A, omega, phi, C):
        return A * np.sin(omega * x + phi) + C
    
    x = np.arange(len(blue_sequence))
    y = np.array(blue_sequence)
    
    A_guess = (np.max(y) - np.min(y)) / 2
    C_guess = np.mean(y)
    omega_guess = 2 * np.pi / 9
    
    try:
        params, _ = curve_fit(
            sine_func, x, y,
            p0=[A_guess, omega_guess, 0, C_guess],
            bounds=([0, 2*np.pi/15, -np.pi, 1],
                    [8, 2*np.pi/5, np.pi, 16]),
            maxfev=2000
        )
        A, omega, phi, C = params
        pred_val = sine_func(len(blue_sequence), A, omega, phi, C)
        pred = int(round(pred_val))
        return max(1, min(16, pred))
    except:
        return int(round(np.mean(blue_sequence)))


def get_blue_sine_prediction(draws: List[Dict], window: int = 8) -> Tuple[int, List[int], List[int]]:
    """
    获取蓝球正弦拟合预测数据（供绘图使用）
    
    参数:
        draws: 历史开奖数据
        window: 正弦拟合窗口（默认8期）
    返回:
        (预测值, 最近window期蓝球列表, 对应期号列表)
    """
    if len(draws) < window:
        return 8, [], []
    
    recent_blues = []
    recent_periods = []
    for draw in draws[-window:]:
        blue = draw.get('blue', 0)
        if 1 <= blue <= 16:
            recent_blues.append(blue)
            recent_periods.append(draw.get('period', ''))
    
    if len(recent_blues) < 6:
        return 8, recent_blues, recent_periods
    
    prediction = sine_fit_predict_blue(recent_blues)
    
    return prediction, recent_blues, recent_periods


def get_blue_series_for_plot(draws: List[Dict], lookback: int = 100) -> Tuple[List[int], List[str]]:
    """
    获取最近lookback期蓝球序列（供绘图使用）
    
    参数:
        draws: 历史开奖数据
        lookback: 回溯期数（默认100期）
    返回:
        (蓝球序列, 期号序列)
    """
    if len(draws) > lookback:
        recent_draws = draws[-lookback:]
    else:
        recent_draws = draws
    
    blue_series = []
    period_series = []
    for draw in recent_draws:
        blue = draw.get('blue', 0)
        if 1 <= blue <= 16:
            blue_series.append(blue)
            period_series.append(str(draw.get('period', '')))
    
    return blue_series, period_series


# ==================== 正弦拟合和值预测（供绘图使用） ====================

def sine_fit_predict_sum(recent_sums: List[int]) -> int:
    """
    正弦拟合预测下一期和值（纯函数）
    
    参数:
        recent_sums: 最近10期和值列表
    返回:
        预测的和值（80-125范围内）
    """
    from scipy.optimize import curve_fit
    
    if len(recent_sums) < 10:
        return int(round(np.mean(recent_sums))) if recent_sums else 102
    
    def sine_func(x, A, omega, phi, C):
        return A * np.sin(omega * x + phi) + C
    
    x = np.arange(len(recent_sums))
    y = np.array(recent_sums)
    
    A_guess = (np.max(y) - np.min(y)) / 2
    C_guess = np.mean(y)
    omega_guess = 2 * np.pi / 6.5
    
    try:
        params, _ = curve_fit(
            sine_func, x, y,
            p0=[A_guess, omega_guess, 0, C_guess],
            bounds=([0, 2*np.pi/15, -np.pi, 80], 
                    [50, 2*np.pi/4, np.pi, 125]),
            maxfev=2000
        )
        A, omega, phi, C = params
        next_val = sine_func(len(recent_sums), A, omega, phi, C)
        return max(80, min(125, int(round(next_val))))
    except:
        return int(round(np.mean(recent_sums)))


# ==================== 动态和值预测（保留原函数，以防其他地方调用） ====================

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
    
    if abs(deviation) > 15:
        target = RED_EXPECTED_SUM + int(deviation * 0.3)
    else:
        target = RED_EXPECTED_SUM
    
    target = max(79, min(125, target))
    
    return int(target), RED_SUM_STD


# ==================== ML信号分析 ====================
def calculate_ml_signals(draws: List[Dict]) -> Dict:
    """
    计算ML特征信号（固定50期窗口版本 v15.0）
    
    关键修改：固定使用最近50期数据，与选号方法无关
    包含：奖池分析、剪刀差、周期预测、动态注数等
    """
    # 固定使用最近50期数据
    if len(draws) < 10:
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
    
    # 固定使用最近50期数据（关键修改！）
    if len(draws) > 50:
        recent_draws = draws[-50:]
    else:
        recent_draws = draws
    
    latest = recent_draws[-1]
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
    
    # 2. 剪刀差信号（基于50期内数据）
    if len(recent_draws) >= 2:
        prev = recent_draws[-2]
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
    
    # 3. 头奖周期预测（基于50期内数据）
    recent_prizes = [d.get('prize1_count', 0) for d in recent_draws[-20:]]
    high_prize_count = sum(1 for p in recent_prizes if p >= 10)
    
    # 计算距离上次爆发的期数（在50期内搜索）
    last_burst = None
    for idx, d in enumerate(reversed(recent_draws)):
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
    
    # 5. 蓝球偏向预测（基于50期内数据）
    recent_blues = [d.get('blue', 0) for d in recent_draws[-20:] if d.get('blue', 0) > 0]
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
    
    # 6. 红球重复预测（基于50期内数据）
    last_reds = recent_draws[-1].get('reds', [])
    prev_reds = recent_draws[-2].get('reds', []) if len(recent_draws) >= 2 else []
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
    
    recommended_bets = max(0, min(calculated_bets, max_bets_by_signal))
    
    reasons = [base_reason]
    if scissors_hint:
        reasons.append(scissors_hint)
    if cycle_hint:
        reasons.append(cycle_hint)
    reasons.append(f"信号{signal_strength}分{max_bets_by_signal}组上限")
    reason_text = " → ".join(reasons)
    
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


# ==================== 原方法1：冷热码评分 + 和值动态预测 ====================

# ============================================================
# 方法2：胆拖混合（基于新系统评分的简化版）
# 胆码 = 评分Top5中随机抽2个
# 剩余4个 = 其余28个中按Softmax抽取
# ============================================================

class Method2DanTuo:
    """
    方法2：胆拖混合（新实现 v15.0）
    
    核心逻辑：
    1. 使用新规则系统的评分
    2. 胆码从评分Top5中随机抽取2个
    3. 剩余4个从其余28个中按Softmax概率抽取
    4. 使用正弦拟合和值筛选
    5. 动态降级策略（5+1 → 4+2 → 保底）
    """
    #-----------------
    def __init__(self, draws: List[Dict], sum_method: str = None, blue_method: str = None):
        # 固定使用最近100期数据
        if len(draws) > 100:
            self.draws = draws[-100:]
        else:
            self.draws = draws
        
        # 使用新规则系统的评分计算器（传递 sum_method）
        self.scorer = Method1NewRule(draws, sum_method=sum_method)
        
        # 可调节参数
        self.normal_threshold = 50
        self.temp_normal = 0.8
        self.sum_tolerance = 12
        self.require_consecutive = True
        self.max_attempts = 500
        
        # 存储蓝球预测方法
        self.blue_method = blue_method
        
        # 从scorer获取评分和池子
        self.red_scores = self.scorer.red_scores
        self.normal_pool = self.scorer.normal_pool
        self.cold_pool = self.scorer.cold_pool
    
    def _softmax_select(self, pool: List[int], scores: Dict[int, int], temperature: float) -> int:
        """Softmax概率抽取"""
        if not pool:
            return None
        score_list = [scores[num] for num in pool]
        exp_scores = np.exp(np.array(score_list) / temperature)
        probs = exp_scores / np.sum(exp_scores)
        return np.random.choice(pool, p=probs)
    
    def _get_top_n_by_score(self, n: int) -> List[int]:
        """获取评分最高的N个号码"""
        sorted_nums = sorted(self.red_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_nums[:n]]
    #-----
    def _get_target_sum(self) -> Tuple[int, int]:
        import streamlit as st
        import traceback
        
        try:
            sum_method = st.session_state.get('sum_predict_method', '7期均值')
            
            if sum_method == "正弦拟合":
                target = generate_target_sum_by_sine(self.draws)
            else:
                target = generate_target_sum_by_range(self.draws)
            
            return target, self.sum_tolerance
        except Exception as e:
            st.error(f"_get_target_sum 错误: {e}")
            st.code(traceback.format_exc())
            return 102, self.sum_tolerance
    
    def _has_consecutive(self, reds: List[int]) -> bool:
        """检查是否有连号"""
        for i in range(1, len(reds)):
            if reds[i] - reds[i-1] == 1:
                return True
        return False
    
    def _is_valid(self, reds: List[int], target_sum: int) -> bool:
        """后置筛选：和值 + 连号"""
        total = sum(reds)
        if abs(total - target_sum) > self.sum_tolerance:
            return False
        if self.require_consecutive and not self._has_consecutive(reds):
            return False
        return True
    
    def _expand_to_7(self, reds_6: List[int]) -> List[int]:
        """将6个红球扩展为7个（从正常池补充）"""
        candidates = [n for n in self.normal_pool if n not in reds_6]
        if not candidates:
            candidates = [n for n in RED_NUMBERS if n not in reds_6]
        if candidates:
            extra = self._softmax_select(candidates, self.red_scores, self.temp_normal)
            if extra is not None:
                return sorted(reds_6 + [extra])
        return sorted(reds_6 + [np.random.choice([n for n in RED_NUMBERS if n not in reds_6])])
    
    def _sample_with_dantuo(self, normal_count: int, cold_count: int) -> List[int]:
        """
        胆拖混合抽样
        胆码：从评分Top5中随机抽2个
        剩余：从其余号码中按Softmax抽 normal_count + cold_count - 2 个
        """
        # 1. 获取评分Top5作为胆码候选池
        top5 = self._get_top_n_by_score(5)
        if len(top5) < 2:
            # 数据不足，降级到普通分层抽样
            return self._sample_stratified(normal_count, cold_count)
        
        # 2. 随机抽取2个胆码
        anchors = np.random.choice(top5, size=2, replace=False).tolist()
        
        # 3. 剩余号码池（排除胆码）
        remaining_pool = [n for n in RED_NUMBERS if n not in anchors]
        
        # 4. 需要抽取的数量
        needed = normal_count + cold_count - 2
        
        if needed <= 0 or len(remaining_pool) < needed:
            return None
        
        # 5. 按Softmax从剩余池中抽取
        selected = anchors.copy()
        temp_pool = remaining_pool.copy()
        
        for _ in range(needed):
            if not temp_pool:
                break
            # 根据号码所在池子选择温度
            num = temp_pool[0]
            if self.red_scores.get(num, 0) >= self.normal_threshold:
                temp = 0.8
            else:
                temp = 1.2
            num = self._softmax_select(temp_pool, self.red_scores, temp)
            if num is not None:
                selected.append(num)
                temp_pool.remove(num)
        
        selected = list(set(selected))
        if len(selected) < 6:
            return None
        
        return sorted(selected[:6])
    
    def _sample_stratified(self, normal_count: int, cold_count: int) -> List[int]:
        """
        分层抽样（降级备用）
        正常池和冷码池分别抽取
        """
        selected = []
        
        # 从正常池抽取
        temp_normal = self.normal_pool.copy()
        for _ in range(min(normal_count, len(temp_normal))):
            if not temp_normal:
                break
            num = self._softmax_select(temp_normal, self.red_scores, 0.8)
            if num is not None:
                selected.append(num)
                temp_normal.remove(num)
        
        # 从冷码池抽取
        temp_cold = self.cold_pool.copy()
        for _ in range(min(cold_count, len(temp_cold))):
            if not temp_cold:
                break
            num = self._softmax_select(temp_cold, self.red_scores, 1.2)
            if num is not None:
                selected.append(num)
                temp_cold.remove(num)
        
        selected = list(set(selected))
        if len(selected) < 6:
            return None
        return sorted(selected[:6])
    #-------
    def generate_bets(self, num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
        """
        生成投注（核心方法）
        动态降级策略：5+1 → 4+2 → 保底
        """
        bets = []
        
        for _ in range(num_bets):
            # ✅ 每注独立生成和值目标
            target_sum, _ = self._get_target_sum()
            
            # 使用 target_sum 进行筛选
            success = False
            
            # ========== 第1层：5+1（胆拖模式） ==========
            for _ in range(self.max_attempts):
                selected = self._sample_with_dantuo(5, 1)
                if selected and self._is_valid(selected, target_sum):
                    final_reds = self._expand_to_7(selected)
                    bets.append({
                        'reds': final_reds,
                        'blues': [8],  # 蓝球暂用默认值
                        'blue': 8,
                        'sum': sum(final_reds),
                        'method': '方法2:胆拖混合(5+1)'
                    })
                    success = True
                    break
            
            if success:
                continue
            
            # ========== 第2层：4+2（放宽容差） ==========
            original_tolerance = self.sum_tolerance
            self.sum_tolerance = min(20, int(original_tolerance * 1.5))
            
            for _ in range(self.max_attempts // 2):
                selected = self._sample_with_dantuo(4, 2)
                if selected and self._is_valid(selected, target_sum):
                    final_reds = self._expand_to_7(selected)
                    bets.append({
                        'reds': final_reds,
                        'blues': [8],
                        'blue': 8,
                        'sum': sum(final_reds),
                        'method': '方法2:胆拖混合(4+2放宽和值)'
                    })
                    success = True
                    break
            
            self.sum_tolerance = original_tolerance
            
            if success:
                continue
            
            # ========== 第3层：放弃和值，只保留连号 ==========
            for _ in range(self.max_attempts // 2):
                selected = self._sample_with_dantuo(4, 2)
                if selected and self._has_consecutive(selected):
                    final_reds = self._expand_to_7(selected)
                    bets.append({
                        'reds': final_reds,
                        'blues': [8],
                        'blue': 8,
                        'sum': sum(final_reds),
                        'method': '方法2:胆拖混合(仅连号)'
                    })
                    success = True
                    break
            
            if success:
                continue
            
            # ========== 第4层：保底随机 ==========
            selected = sorted(np.random.choice(RED_NUMBERS, size=6, replace=False))
            final_reds = self._expand_to_7(selected)
            bets.append({
                'reds': final_reds,
                'blues': [8],
                'blue': 8,
                'sum': sum(final_reds),
                'method': '方法2:胆拖混合(保底)'
            })
        
        return bets

#-------
# ============================================================
# 方法3：LightGBM（完整版）
# 修改内容：
# 1. 降级逻辑改为降级到方法2（新实现）
# 2. 蓝球评分改用新系统
# 3. 其他ML训练和预测逻辑保持不变
# ============================================================

class Method3LightGBM:
    """方法3：LightGBM梯度提升树 + 规律特征"""
    #------------
    def __init__(self, draws: List[Dict], use_cache: bool = True, sum_method: str = None, blue_method: str = None):
        # 固定使用最近100期数据
        if len(draws) > 100:
            self.draws = draws[-100:]
        else:
            self.draws = draws
        self.use_cache = use_cache
        self.model = None
        self.is_trained = False
        # 存储预测方法（备用）
        self.sum_method = sum_method
        self.blue_method = blue_method
    
    def _extract_features(self, window_draws: List[Dict], target_num: int) -> Optional[Dict]:
        """提取特征（与原版相同）"""
        if len(window_draws) < 20:
            return None
        
        features = {}
        total = len(window_draws)
        
        # 频率特征
        freq = sum(1 for d in window_draws if target_num in d.get('reds', []))
        features['freq'] = freq / total if total > 0 else 0
        
        # 遗漏特征
        last_seen = None
        for idx, d in enumerate(reversed(window_draws)):
            if target_num in d.get('reds', []):
                last_seen = idx
                break
        features['absence'] = last_seen if last_seen is not None else total
        
        # 近期频率
        recent_10 = window_draws[-10:] if len(window_draws) >= 10 else window_draws
        recent_20 = window_draws[-20:] if len(window_draws) >= 20 else window_draws
        features['recent_freq_10'] = sum(1 for d in recent_10 if target_num in d.get('reds', [])) / max(len(recent_10), 1)
        features['recent_freq_20'] = sum(1 for d in recent_20 if target_num in d.get('reds', [])) / max(len(recent_20), 1)
        
        # 上期出现
        if window_draws:
            features['last_appeared'] = 1 if target_num in window_draws[-1].get('reds', []) else 0
        
        # 分区、奇偶、大小
        zone = (target_num - 1) // 11 + 1
        features['zone'] = zone
        features['parity'] = target_num % 2
        features['size'] = 0 if target_num <= 16 else 1
        
        # 规律特征
        last_draw = window_draws[-1] if window_draws else {}
        last_reds = last_draw.get('reds', [])
        
        if last_reds:
            last_reds_sorted = sorted(last_reds)
            features['is_repeat'] = 1 if target_num in last_reds else 0
            min_edge_dist = min(abs(target_num - r) for r in last_reds)
            features['is_edge'] = 1 if min_edge_dist == 1 else 0
            features['min_edge_distance'] = min_edge_dist
            
            # 夹号特征
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
            
            # 连号特征
            test_reds = sorted(set(last_reds_sorted) | {target_num})
            features['consecutive_length'] = self._calculate_consecutive_length(test_reds, target_num)
            
            # 对称号
            symmetric = 34 - target_num
            features['symmetric_in_last'] = 1 if symmetric in last_reds else 0
            
            # 蓝球相关
            last_blue = last_draw.get('blue', 0)
            features['blue_diff'] = abs(target_num - last_blue) if last_blue > 0 else 99
            features['blue_same_parity'] = 1 if (target_num % 2) == (last_blue % 2) else 0
        
        # 历史规律率
        features['edge_historical_rate'] = self._calc_edge_historical_rate(window_draws, target_num)
        features['gap_historical_rate'] = self._calc_gap_historical_rate(window_draws, target_num)
        
        return features
    
    def _calculate_consecutive_length(self, reds: List[int], target_num: int) -> int:
        """计算连号长度（与原版相同）"""
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
    
    def _calc_edge_historical_rate(self, draws: List[Dict], target_num: int) -> float:
        """计算边号历史开出率（与原版相同）"""
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
        """计算夹号历史开出率（与原版相同）"""
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
        """训练LightGBM模型（与原版相同）"""
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
        """预测Top N红球（与原版相同）"""
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
    
    def _calculate_blue_scores(self) -> Dict[int, float]:
        """计算蓝球评分（使用新系统的简化版）"""
        # 使用新蓝球系统的评分逻辑
        try:
            blue_system = BlueScoreSystem(self.draws)
            scores = {}
            for num in range(1, 17):
                scores[num] = blue_system.calculate_total_score(num)
            return scores
        except:
            # 降级：使用频率评分
            freq = {i: 0 for i in range(1, 17)}
            for draw in self.draws[-100:]:
                blue = draw.get('blue', 0)
                if 1 <= blue <= 16:
                    freq[blue] += 1
            max_freq = max(freq.values()) if freq.values() else 1
            return {num: cnt / max_freq for num, cnt in freq.items()}
    
    def _get_top_blues(self, n: int = 3) -> List[int]:
        """获取评分最高的N个蓝球"""
        blue_scores = self._calculate_blue_scores()
        sorted_blues = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_blues[:n]]
    
    def generate_bets(self, num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
        """
        生成投注
        修改：训练失败时降级到方法2（新实现）而非方法1
        """
        if not self.is_trained:
            self.train()
        
        if not self.is_trained:
            # 降级到方法2（新实现的胆拖混合）
            fallback = Method2DanTuo(self.draws)
            return fallback.generate_bets(num_bets, bet_type)
        
        # ML预测逻辑
        top_reds = self.predict_top_reds(12)
        if len(top_reds) < 6:
            top_reds = list(range(1, 34))
            random.shuffle(top_reds)
        
        # 蓝球评分（使用新系统）
        blue_scores = self._calculate_blue_scores()
        blue_weights = np.array([math.exp(blue_scores.get(i, 0)) for i in BLUE_NUMBERS])
        if np.sum(blue_weights) > 0:
            blue_weights = blue_weights / np.sum(blue_weights)
        else:
            blue_weights = np.ones(16) / 16
        
        red_count = 7 if bet_type.startswith("7") else 8
        blue_count = 2 if bet_type == "7+2" else 1
        
        # 获取Top蓝球（用于7+2）
        top_blues = self._get_top_blues(3)
        
        bets = []
        for _ in range(num_bets):
            # ML选择核心6码
            core_reds = top_reds[:6].copy()
            random.shuffle(core_reds)
            core_reds = sorted(core_reds[:6])
            
            # 扩展为7码或8码
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
            
            # 蓝球选择
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
                'method': '方法3:LightGBM+规律特征'
            })
        
        return bets

#--------------
# ============================================================
# 方法4：XGBoost（完整版）
# 修改内容：
# 1. 降级逻辑改为降级到方法2（新实现）
# 2. 蓝球评分改用新系统
# 3. 其他ML训练和预测逻辑保持不变
# ============================================================

class Method4Ensemble:
    """方法4：XGBoost + 规律特征"""
    #---------------
    def __init__(self, draws: List[Dict], use_cache: bool = True, sum_method: str = None, blue_method: str = None):
        # 固定使用最近120期数据
        if len(draws) > 120:
            self.draws = draws[-120:]
        else:
            self.draws = draws
        self.use_cache = use_cache
        self.xgb_model = None
        self.is_trained = False
        # 存储预测方法（备用）
        self.sum_method = sum_method
        self.blue_method = blue_method
    
    def _extract_features(self, window_draws: List[Dict], target_num: int) -> Optional[Dict]:
        """提取特征（与原版相同）"""
        if len(window_draws) < 20:
            return None
        
        features = {}
        total = len(window_draws)
        
        # 频率特征
        freq = sum(1 for d in window_draws if target_num in d.get('reds', []))
        features['freq'] = freq / total if total > 0 else 0
        
        # 遗漏特征
        last_seen = None
        for idx, d in enumerate(reversed(window_draws)):
            if target_num in d.get('reds', []):
                last_seen = idx
                break
        features['absence'] = last_seen if last_seen is not None else total
        
        # 近期频率
        recent_10 = window_draws[-10:] if len(window_draws) >= 10 else window_draws
        recent_20 = window_draws[-20:] if len(window_draws) >= 20 else window_draws
        features['recent_freq_10'] = sum(1 for d in recent_10 if target_num in d.get('reds', [])) / max(len(recent_10), 1)
        features['recent_freq_20'] = sum(1 for d in recent_20 if target_num in d.get('reds', [])) / max(len(recent_20), 1)
        
        # 上期出现
        if window_draws:
            features['last_appeared'] = 1 if target_num in window_draws[-1].get('reds', []) else 0
        
        # 分区、奇偶、大小
        zone = (target_num - 1) // 11 + 1
        features['zone'] = zone
        features['parity'] = target_num % 2
        features['size'] = 0 if target_num <= 16 else 1
        
        # 规律特征
        last_draw = window_draws[-1] if window_draws else {}
        last_reds = last_draw.get('reds', [])
        
        if last_reds:
            last_reds_sorted = sorted(last_reds)
            features['is_repeat'] = 1 if target_num in last_reds else 0
            min_edge_dist = min(abs(target_num - r) for r in last_reds)
            features['is_edge'] = 1 if min_edge_dist == 1 else 0
            features['min_edge_distance'] = min_edge_dist
            
            # 夹号特征
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
            
            # 连号特征
            test_reds = sorted(set(last_reds_sorted) | {target_num})
            features['consecutive_length'] = self._calculate_consecutive_length(test_reds, target_num)
            
            # 对称号
            symmetric = 34 - target_num
            features['symmetric_in_last'] = 1 if symmetric in last_reds else 0
            
            # 蓝球相关
            last_blue = last_draw.get('blue', 0)
            features['blue_diff'] = abs(target_num - last_blue) if last_blue > 0 else 99
            features['blue_same_parity'] = 1 if (target_num % 2) == (last_blue % 2) else 0
        
        # 历史规律率
        features['edge_historical_rate'] = self._calc_edge_historical_rate(window_draws, target_num)
        features['gap_historical_rate'] = self._calc_gap_historical_rate(window_draws, target_num)
        
        return features
    
    def _calculate_consecutive_length(self, reds: List[int], target_num: int) -> int:
        """计算连号长度（与原版相同）"""
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
    
    def _calc_edge_historical_rate(self, draws: List[Dict], target_num: int) -> float:
        """计算边号历史开出率（与原版相同）"""
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
        """计算夹号历史开出率（与原版相同）"""
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
        """训练XGBoost模型（与原版相同）"""
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
        """预测Top N红球（与原版相同）"""
        if not self.is_trained or self.xgb_model is None:
            # 降级到方法2
            fallback = Method2DanTuo(self.draws)
            top_reds = fallback._get_top_n_by_score(n) if hasattr(fallback, '_get_top_n_by_score') else list(range(1, 34))
            return top_reds[:n] if len(top_reds) >= n else top_reds + list(range(1, 34))[:n-len(top_reds)]
        
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
    
    def _calculate_blue_scores(self) -> Dict[int, float]:
        """计算蓝球评分（使用新系统的简化版）"""
        try:
            blue_system = BlueScoreSystem(self.draws)
            scores = {}
            for num in range(1, 17):
                scores[num] = blue_system.calculate_total_score(num)
            return scores
        except:
            # 降级：使用频率评分
            freq = {i: 0 for i in range(1, 17)}
            for draw in self.draws[-100:]:
                blue = draw.get('blue', 0)
                if 1 <= blue <= 16:
                    freq[blue] += 1
            max_freq = max(freq.values()) if freq.values() else 1
            return {num: cnt / max_freq for num, cnt in freq.items()}
    
    def _get_top_blues(self, n: int = 3) -> List[int]:
        """获取评分最高的N个蓝球"""
        blue_scores = self._calculate_blue_scores()
        sorted_blues = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)
        return [num for num, _ in sorted_blues[:n]]
    
    def generate_bets(self, num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
        """
        生成投注
        修改：训练失败时降级到方法2（新实现）而非方法1
        """
        if not self.is_trained:
            self.train()
        
        if not self.is_trained or self.xgb_model is None:
            # 降级到方法2（新实现的胆拖混合）
            fallback = Method2DanTuo(self.draws)
            return fallback.generate_bets(num_bets, bet_type)
        
        # ML预测逻辑
        top_reds = self.predict_top_reds(12)
        if len(top_reds) < 6:
            top_reds = list(range(1, 34))
            random.shuffle(top_reds)
        
        # 蓝球评分（使用新系统）
        blue_scores = self._calculate_blue_scores()
        blue_weights = np.array([math.exp(blue_scores.get(i, 0)) for i in BLUE_NUMBERS])
        if np.sum(blue_weights) > 0:
            blue_weights = blue_weights / np.sum(blue_weights)
        else:
            blue_weights = np.ones(16) / 16
        
        red_count = 7 if bet_type.startswith("7") else 8
        blue_count = 2 if bet_type == "7+2" else 1
        
        # 获取Top蓝球（用于7+2）
        top_blues = self._get_top_blues(3)
        
        bets = []
        for _ in range(num_bets):
            # ML选择核心6码
            core_reds = top_reds[:6].copy()
            random.shuffle(core_reds)
            core_reds = sorted(core_reds[:6])
            
            # 扩展为7码或8码
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
            
            # 蓝球选择
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
                'method': '方法4:XGBoost+规律特征'
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
#---------------
# ============================================================
# 新规则系统 v15.0 - Method1NewRule
# 替代原方法1，用于红球选号
# ============================================================

class Method1NewRule:
    """
    新规则系统 v15.0
    核心特性：单号码评分 + 分池 + 动态降级 + 正弦拟合和值 + 后置筛选
    """
    #-----------------
    def __init__(self, draws: List[Dict], sum_method: str = None):
        # ========== 固定使用最近100期数据 ==========
        if len(draws) > 100:
            self.draws = draws[-100:]
        else:
            self.draws = draws
        
        self.red_scores = None
        self.normal_pool = None
        self.cold_pool = None
        
        # 可调节参数（支持外部修改）
        self.normal_threshold = 50
        self.normal_count_layer1 = 5
        self.cold_count_layer1 = 1
        self.normal_count_layer2 = 4
        self.cold_count_layer2 = 2
        self.temp_normal = 0.8
        self.temp_cold = 1.2
        self.sum_tolerance = 12
        self.require_consecutive = True
        self.max_attempts_layer1 = 500
        self.max_attempts_layer2 = 300
    
        # ========== 加分项开关（默认全部开启） ==========
        self.enable_freq_acc = True
        self.enable_density_trend = True
        self.enable_absence_bonus = True
        self.enable_alternating = True
        
        # ========== 存储预测方法 ==========
        self.sum_method = sum_method
        
        # 计算评分和分池
        self._calculate_scores_and_pools()
    
    def _calculate_scores_and_pools(self):
        """计算所有红球的综合评分并分池"""
        self.red_scores = {}
        for num in RED_NUMBERS:
            self.red_scores[num] = self._calculate_total_score(num)
        
        # 分池
        self.normal_pool = [num for num in RED_NUMBERS if self.red_scores[num] >= self.normal_threshold]
        self.cold_pool = [num for num in RED_NUMBERS if self.red_scores[num] < self.normal_threshold]
    #---
    def _calculate_total_score(self, num: int) -> int:
        """
        计算单号码综合评分
        公式：基础分 + 可选加分项（基础分始终开启）
        """
        import streamlit as st
        
        absence = self._calculate_absence(num)
        
        # 基础分（始终开启）
        base_score = self._get_base_score(absence)
        
        # 如果是上期号码，直接返回基础分
        if absence == 0:
            return base_score
        
        bonus = 0
            
        # 1. 频率加速度
        freq_acc = self._calculate_frequency_acceleration(num)
        if self.enable_freq_acc and freq_acc > 0.1:
            bonus += st.session_state.get('bonus_freq_acc', 25)
            # 统计触发次数
            if not hasattr(self, '_freq_acc_count'):
                self._freq_acc_count = 0
            self._freq_acc_count += 1
        
        # 2. 疏转密
        if self.enable_density_trend and self._is_density_turning(num):
            bonus += st.session_state.get('bonus_density_trend', 20)
            if not hasattr(self, '_density_count'):
                self._density_count = 0
            self._density_count += 1
        
        # 3. 遗漏13-20期
        if self.enable_absence_bonus and 13 <= absence <= 20:
            bonus += st.session_state.get('bonus_absence', 30)
            if not hasattr(self, '_absence_count'):
                self._absence_count = 0
            self._absence_count += 1
        
        # 4. 隔期模式
        if self.enable_alternating and self._is_alternating(num):
            bonus += st.session_state.get('bonus_alternating', 12)
            if not hasattr(self, '_alt_count'):
                self._alt_count = 0
            self._alt_count += 1
        
        return base_score + bonus
    
    def _calculate_absence(self, num: int) -> int:
        """计算遗漏期数"""
        absence = 0
        for draw in reversed(self.draws):
            if num in draw.get('reds', []):
                break
            absence += 1
        return absence
    #----
    def _get_base_score(self, absence: int) -> int:
        """基础分（基于遗漏值，从用户设置读取）"""
        import streamlit as st
        
        if absence == 0:
            return st.session_state.get('score_0', 70)
        elif 1 <= absence <= 5:
            return st.session_state.get('score_1_5', 50)
        elif 6 <= absence <= 10:
            return st.session_state.get('score_6_10', 20)
        elif 11 <= absence <= 15:
            return st.session_state.get('score_11_15', 15)
        elif 16 <= absence <= 20:
            return st.session_state.get('score_16_20', 10)
        elif 21 <= absence <= 30:
            return st.session_state.get('score_21_30', 8)
        else:
            return st.session_state.get('score_30_plus', 5)
    
    def _calculate_frequency_acceleration(self, num: int) -> float:
        """计算频率加速度 Δf@(20→5)"""
        if len(self.draws) < 20:
            return 0
        
        recent_5 = self.draws[-5:]
        recent_20 = self.draws[-20:]
        
        count_5 = sum(1 for d in recent_5 if num in d.get('reds', []))
        count_20 = sum(1 for d in recent_20 if num in d.get('reds', []))
        
        return (count_5 / 5) - (count_20 / 20)
    
    def _is_density_turning(self, num: int) -> bool:
        """判断是否处于疏转密状态"""
        if len(self.draws) < 50:
            return False
        
        half = 25
        recent = self.draws[-half:]
        earlier = self.draws[-50:-half]
        
        recent_density = sum(1 for d in recent if num in d.get('reds', [])) / len(recent)
        earlier_density = sum(1 for d in earlier if num in d.get('reds', [])) / len(earlier) if earlier else 0
        
        trend = recent_density - earlier_density
        
        recent_count = sum(1 for d in self.draws[-5:] if num in d.get('reds', []))
        
        return trend > 0.12 and recent_count >= 2
    
    def _is_alternating(self, num: int) -> bool:
        """判断是否为隔期模式（间隔恰好2期）"""
        positions = [i for i, draw in enumerate(self.draws) if num in draw.get('reds', [])]
        if len(positions) < 2:
            return False
        last_gap = positions[-1] - positions[-2]
        return last_gap == 2
    #----------------
    def _softmax_select(self, pool: List[int], scores: Dict[int, int], temperature: float = None) -> int:
    """
    线性概率抽取（概率 = 分数 × 0.0012，自动归一化）
    temperature 参数保留是为了兼容性，实际不使用
    """
    if not pool:
        return None
    score_list = np.array([max(1, scores[num]) for num in pool])
    # 赋以概率 = 分数 × 0.0012
    raw_probs = score_list * 0.0012
    # 归一化
    probs = raw_probs / np.sum(raw_probs)
    selected = np.random.choice(pool, p=probs)
    
    # ========== 调试：打印选中号码及其分数 ==========
    print(f"[选中] 号码: {selected:02d}, 分数: {scores[selected]}, 池子大小: {len(pool)}")
    # ============================================
    
    return selected
    
    def _has_consecutive(self, reds: List[int]) -> bool:
        """检查是否有连号"""
        for i in range(1, len(reds)):
            if reds[i] - reds[i-1] == 1:
                return True
        return False
    
    def _is_valid(self, reds: List[int], target_sum: int) -> bool:
        """后置筛选：和值 + 连号"""
        total = sum(reds)
        if abs(total - target_sum) > self.sum_tolerance:
            return False
        if self.require_consecutive and not self._has_consecutive(reds):
            return False
        return True
    
    def _get_target_sum(self):
        """
        根据用户选择的预测方法返回和值目标（每注独立随机）
        """
        import streamlit as st
        
        sum_method = st.session_state.get('sum_predict_method', '7期均值')
        
        if sum_method == "正弦拟合":
            target = generate_target_sum_by_sine(self.draws)
        else:
            target = generate_target_sum_by_range(self.draws)
        
        return target, self.sum_tolerance
    
    def _sample_reds(self, normal_count: int, cold_count: int) -> List[int]:
        # 只打印一次
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            scores = list(self.red_scores.values())
            print(f"=== 分数分布 ===")
            print(f"最高分: {max(scores)}")
            print(f"最低分: {min(scores)}")
            print(f"平均分: {sum(scores)/len(scores):.1f}")
            print(f"正常池大小: {len(self.normal_pool)}")
            print(f"冷码池大小: {len(self.cold_pool)}")
        """分层抽取指定数量的红球"""
        selected = []
        
        temp_normal = self.normal_pool.copy()
        for _ in range(min(normal_count, len(temp_normal))):
            if not temp_normal:
                break
            num = self._softmax_select(temp_normal, self.red_scores, self.temp_normal)
            if num is not None:
                selected.append(num)
                temp_normal.remove(num)
        
        temp_cold = self.cold_pool.copy()
        for _ in range(min(cold_count, len(temp_cold))):
            if not temp_cold:
                break
            num = self._softmax_select(temp_cold, self.red_scores, self.temp_cold)
            if num is not None:
                selected.append(num)
                temp_cold.remove(num)
        
        selected = list(set(selected))
        if len(selected) < 6:
            return None
        return sorted(selected[:6])
    
    def _expand_to_7(self, reds_6: List[int]) -> List[int]:
        """将6个红球扩展为7个（从正常池补充）"""
        candidates = [n for n in self.normal_pool if n not in reds_6]
        if not candidates:
            candidates = [n for n in RED_NUMBERS if n not in reds_6]
        if candidates:
            extra = self._softmax_select(candidates, self.red_scores, self.temp_normal)
            if extra is not None:
                return sorted(reds_6 + [extra])
        return sorted(reds_6 + [np.random.choice([n for n in RED_NUMBERS if n not in reds_6])])
    
    def generate_bets(self, num_bets: int = 4, bet_type: str = "7+1") -> List[Dict]:
        """
        生成投注（核心方法）
        动态降级策略：5+1 → 4+2 → 保底
        """
        bets = []
        
        for _ in range(num_bets):
            # 每注独立生成和值目标
            target_sum, _ = self._get_target_sum()
            success = False
            
            # ========== 第1层：5+1 ==========
            for _ in range(self.max_attempts_layer1):
                selected = self._sample_reds(self.normal_count_layer1, self.cold_count_layer1)
                if selected and self._is_valid(selected, target_sum):
                    final_reds = self._expand_to_7(selected)
                    bets.append({
                        'reds': final_reds,
                        'blues': [8],
                        'blue': 8,
                        'sum': sum(final_reds),
                        'method': '新规则系统 v15.0 (5+1)'
                    })
                    success = True
                    break
            
            if success:
                continue
            
            # ========== 第2层：4+2（放宽容差） ==========
            original_tolerance = self.sum_tolerance
            self.sum_tolerance = min(20, int(original_tolerance * 1.5))
            
            for _ in range(self.max_attempts_layer2):
                selected = self._sample_reds(self.normal_count_layer2, self.cold_count_layer2)
                if selected and self._is_valid(selected, target_sum):
                    final_reds = self._expand_to_7(selected)
                    bets.append({
                        'reds': final_reds,
                        'blues': [8],
                        'blue': 8,
                        'sum': sum(final_reds),
                        'method': '新规则系统 v15.0 (4+2 放宽和值)'
                    })
                    success = True
                    break
            
            self.sum_tolerance = original_tolerance
            
            if success:
                continue
            
            # ========== 第3层：放弃和值，只保留连号 ==========
            for _ in range(self.max_attempts_layer2):
                selected = self._sample_reds(self.normal_count_layer2, self.cold_count_layer2)
                if selected and self._has_consecutive(selected):
                    final_reds = self._expand_to_7(selected)
                    bets.append({
                        'reds': final_reds,
                        'blues': [8],
                        'blue': 8,
                        'sum': sum(final_reds),
                        'method': '新规则系统 v15.0 (仅连号)'
                    })
                    success = True
                    break
            
            if success:
                continue
            
            # ========== 第4层：保底随机 ==========
            selected = sorted(np.random.choice(RED_NUMBERS, size=6, replace=False))
            final_reds = self._expand_to_7(selected)
            bets.append({
                'reds': final_reds,
                'blues': [8],
                'blue': 8,
                'sum': sum(final_reds),
                'method': '新规则系统 v15.0 (保底随机)'
            })
       # 在返回前打印统计
        print(f"=== 加分项触发统计 ===")
        print(f"频率加速度触发次数: {getattr(self, '_freq_acc_count', 0)}")
        print(f"疏转密触发次数: {getattr(self, '_density_count', 0)}")
        print(f"遗漏13-20期触发次数: {getattr(self, '_absence_count', 0)}")
        print(f"隔期模式触发次数: {getattr(self, '_alt_count', 0)}")     
        return bets
#---------------------------
# ============================================================
# 蓝球评分系统 v15.0
# 统一评分制 + Softmax概率抽取
# ============================================================

class BlueScoreSystem:
    """
    蓝球评分系统
    核心特性：多维评分 + 正弦拟合 + Softmax抽取
    """
    #---------
    def __init__(self, draws: List[Dict], blue_method: str = None):
        # 固定使用最近100期数据
        if len(draws) > 100:
            self.draws = draws[-100:]
        else:
            self.draws = draws
        
        # 可调节参数
        self.temperature = 0.8
        self.neighbor1_bonus = 20
        self.neighbor2_bonus = 5
        self.sine_fit_bonus = 20
        self.freq_acc_bonus = 15
        self.mid_absence_bonus = 12
        self.alternating_bonus = 5
        self.remainder_gap4_bonus = 15
        self.remainder_gap6_bonus = 10
        
        # 存储蓝球预测方法
        self.blue_method = blue_method
        
        # 正弦拟合预测值（缓存）
        self._sine_prediction = None
    
    def calculate_absence(self, num: int) -> int:
        """计算蓝球遗漏期数"""
        absence = 0
        for draw in reversed(self.draws):
            if draw.get('blue', 0) == num:
                break
            absence += 1
        return absence
    
    def get_base_score(self, absence: int) -> int:
        """基础分（基于遗漏值）"""
        if absence == 0:
            return 35      # 重号
        elif 1 <= absence <= 4:
            return 50      # 短期热号（最高）
        elif 5 <= absence <= 8:
            return 40      # 中期偏热
        elif 9 <= absence <= 12:
            return 30      # 中等
        elif 13 <= absence <= 16:
            return 20      # 冷热交界
        elif 17 <= absence <= 24:
            return 12      # 偏冷
        elif 25 <= absence <= 32:
            return 8       # 冷
        else:
            return 5       # 极冷
    
    def calculate_frequency_acceleration(self, num: int) -> float:
        """
        计算频率加速度 Δf@(20→10)
        注意：只适用于遗漏≤10的蓝球，遗漏>10时返回0
        """
        absence = self.calculate_absence(num)
        if absence > 10:
            return 0
        
        if len(self.draws) < 20:
            return 0
        
        recent_10 = self.draws[-10:]
        recent_20 = self.draws[-20:]
        
        count_10 = sum(1 for d in recent_10 if d.get('blue', 0) == num)
        count_20 = sum(1 for d in recent_20 if d.get('blue', 0) == num)
        
        return (count_10 / 10) - (count_20 / 20)
    
    def is_alternating(self, num: int) -> bool:
        """判断是否为隔期模式（间隔恰好2期）"""
        positions = [i for i, draw in enumerate(self.draws) if draw.get('blue', 0) == num]
        if len(positions) < 2:
            return False
        last_gap = positions[-1] - positions[-2]
        return last_gap == 2
    
    def get_neighbor_bonus(self, current_blue: int) -> int:
        """计算邻号加分（基于上期蓝球）"""
        if len(self.draws) < 1:
            return 0
        
        last_blue = self.draws[-1].get('blue', 8)
        
        # 计算圆环上的最小距离（1和16相邻）
        diff = min(abs(current_blue - last_blue), 
                   16 - abs(current_blue - last_blue))
        
        if diff == 1:
            return self.neighbor1_bonus
        elif diff == 2:
            return self.neighbor2_bonus
        else:
            return 0
    
    def get_remainder_gap_bonus(self, num: int) -> int:
        """计算除3余数空缺加分"""
        remainder = num % 3
        
        # 计算该余数组最近的空缺期数
        gap = 0
        for draw in reversed(self.draws):
            blue = draw.get('blue', 0)
            if blue % 3 == remainder:
                break
            gap += 1
        
        if 4 <= gap <= 5:
            return self.remainder_gap4_bonus
        elif 6 <= gap <= 7:
            return self.remainder_gap6_bonus
        else:
            return 0
    
    def get_sine_fit_prediction(self) -> int:
        """
        正弦拟合预测下一期蓝球（返回单个预测值）
        使用8期窗口
        """
        if len(self.draws) < 8:
            # 数据不足，返回近期均值
            recent_blues = [d.get('blue', 8) for d in self.draws[-5:] if d.get('blue', 0) > 0]
            if recent_blues:
                return int(round(np.mean(recent_blues)))
            return 8
        
        blue_sequence = [d.get('blue', 0) for d in self.draws[-8:] if d.get('blue', 0) > 0]
        if len(blue_sequence) < 6:
            return int(round(np.mean(blue_sequence)))
        
        try:
            from scipy.optimize import curve_fit
            
            def sine_func(x, A, omega, phi, C):
                return A * np.sin(omega * x + phi) + C
            
            x = np.arange(len(blue_sequence))
            y = np.array(blue_sequence)
            
            A_guess = (np.max(y) - np.min(y)) / 2
            C_guess = np.mean(y)
            omega_guess = 2 * np.pi / 9
            
            params, _ = curve_fit(
                sine_func, x, y,
                p0=[A_guess, omega_guess, 0, C_guess],
                bounds=([0, 2*np.pi/15, -np.pi, 1],
                        [8, 2*np.pi/5, np.pi, 16]),
                maxfev=2000
            )
            A, omega, phi, C = params
            pred_val = sine_func(len(blue_sequence), A, omega, phi, C)
            pred = int(round(pred_val))
            return max(1, min(16, pred))
        except:
            return int(round(np.mean(blue_sequence)))
    
    def get_sine_fit_bonus(self, num: int) -> int:
        """正弦拟合加分：预测值±2范围内+20分"""
        if self._sine_prediction is None:
            self._sine_prediction = self.get_sine_fit_prediction()
        
        pred = self._sine_prediction
        
        # 计算圆环上的最小距离
        diff = min(abs(num - pred), 16 - abs(num - pred))
        
        if diff <= 2:
            return self.sine_fit_bonus
        return 0
    
    def calculate_total_score(self, num: int) -> int:
        """
        计算蓝球综合评分
        公式：基础分 + 各项加分
        """
        absence = self.calculate_absence(num)
        
        # 基础分
        score = self.get_base_score(absence)
        
        # 1. 频率加速度（只适用于遗漏≤10）
        if absence <= 10:
            delta_f = self.calculate_frequency_acceleration(num)
            if delta_f > 0.1:
                score += self.freq_acc_bonus
        
        # 2. 遗漏9-13期加分
        if 9 <= absence <= 13:
            score += self.mid_absence_bonus
        
        # 3. 邻号加分
        score += self.get_neighbor_bonus(num)
        
        # 4. 隔期模式加分
        if self.is_alternating(num):
            score += self.alternating_bonus
        
        # 5. 除3余数空缺加分
        score += self.get_remainder_gap_bonus(num)
        
        # 6. 正弦拟合加分
        score += self.get_sine_fit_bonus(num)
        
        return score
    #--------------
    def select_blue(self) -> int:
        """
        选择蓝球
        根据用户选择的预测方法决定：从候选池中随机抽取
        """
        import streamlit as st
        
        # 从 session_state 获取用户选择的预测方法
        blue_method = st.session_state.get('blue_predict_method', '7期均值')
        
        if blue_method == "正弦拟合":
            # 正弦拟合：从 ±3 环形候选池（7个号码）中随机抽取
            candidates = get_blue_sine_candidates(self.draws, window=8, radius=3)
        else:
            # 7期均值：从 ±3 环形候选池（7个号码）中随机抽取
            candidates = get_blue_range(self.draws, window=7, radius=3)
        
        return random.choice(candidates)


# ============================================================
# 辅助函数：供外部直接调用
# ============================================================

def select_blue_with_new_system(draws: List[Dict]) -> int:
    """
    使用新蓝球系统选择蓝球（快捷函数）
    """
    system = BlueScoreSystem(draws)
    return system.select_blue()

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

#----------------
# ==================== 新规则系统评分获取函数 ====================

def get_red_scores_by_new_system(draws: List[Dict]) -> Dict[int, int]:
    """
    使用新规则系统计算所有红球的综合评分
    固定使用最近100期数据
    """
    if len(draws) < 10:
        return {num: 0 for num in range(1, 34)}
    
    # 固定使用最近100期数据
    if len(draws) > 100:
        recent_draws = draws[-100:]
    else:
        recent_draws = draws
    
    # 创建新规则系统实例并获取评分
    scorer = Method1NewRule(recent_draws)
    return scorer.red_scores


def get_blue_scores_by_new_system(draws: List[Dict]) -> Dict[int, int]:
    """
    使用新蓝球系统计算所有蓝球的综合评分
    固定使用最近100期数据
    """
    if len(draws) < 10:
        return {num: 0 for num in range(1, 17)}
    
    # 固定使用最近100期数据
    if len(draws) > 100:
        recent_draws = draws[-100:]
    else:
        recent_draws = draws
    
    # 创建蓝球评分系统实例
    scorer = BlueScoreSystem(recent_draws)
    
    # 计算每个蓝球的评分
    scores = {}
    for num in range(1, 17):
        scores[num] = scorer.calculate_total_score(num)
    
    return scores
# ============================================================
# 优化版回测函数（支持3种种子模式）
# 修改：方法1改为使用 Method1NewRule（新规则系统）
# ============================================================

def backtest_roi(draws: List[Dict], method_name: str, num_bets: int = 4, lookback: int = 10,
                 seed_mode: str = "date", fixed_seed_value: int = 1,
                 use_ml_signal: bool = False, signal_threshold: int = 50,
                 use_dynamic_bets: bool = True,
                 sum_predict_method: str = '7期均值',
                 blue_predict_method: str = '7期均值') -> Dict:
    """
    优化版ROI回测 - 支持3种种子模式 + ML信号过滤 + 动态投注 + 预测方法选择
    修改：测试最新的 lookback 期，增加中奖明细
    """
    import streamlit as st
    
    method_seed_offset = {
        "方法1": 100,
        "方法2": 200,
        "方法3": 300,
        "方法4": 400,
        "方法5": 500
    }.get(method_name, 0)
    
    train_window = TRAIN_WINDOWS.get(method_name, 100)
    if method_name == "方法5":
        train_window = max(TRAIN_WINDOWS.values())
    
    # 检查数据是否足够
    if len(draws) < train_window + lookback:
        return {
            "roi": 0, "total_cost": 0, "total_prize": 0, "net": 0,
            "win_rate": 0, "periods": 0, "bet_periods": 0, "skip_periods": 0,
            "prize_details": [],
            "error": f"数据不足：需要{train_window + lookback}期，当前{len(draws)}期"
        }
    
    total_cost = 0
    total_prize = 0
    win_count = 0
    bet_periods = 0
    skip_periods = 0
    prize_breakdown = {"first": 0, "second": 0, "third": 0, "fourth": 0, "fifth": 0, "sixth": 0, "fuyun": 0}
    
    # ========== 新增：记录每期中奖明细 ==========
    prize_details = []  # 每个元素: {"period": 期号, "prize": 奖金}
    
    trained_models = {}
    retrain_interval = 5
    
    # 预计算所有期的ML信号（避免重复计算）
    ml_signals_cache = {}
    if use_ml_signal:
        for idx in range(train_window, len(draws)):
            signal_data = draws[:idx]
            ml_signals_cache[idx] = calculate_ml_signals(signal_data)
    
    # ========== 修改：测试最新的 lookback 期 ==========
    start_idx = len(draws) - lookback
    
    for idx in range(lookback):
        i = start_idx + idx
        test_data = draws[i]
        test_period = test_data.get('period', '')
        
        # ========== ML信号强度判断（支持动态投注） ==========
        current_num_bets = num_bets
        
        if use_ml_signal:
            signal_strength = ml_signals_cache.get(i, {}).get('signal_strength', 0)
            
            if use_dynamic_bets:
                if signal_strength >= 60:
                    current_num_bets = 8
                elif signal_strength >= 40:
                    current_num_bets = 4
                elif signal_strength >= 20:
                    current_num_bets = 1
                else:
                    current_num_bets = 0
            else:
                if signal_strength < signal_threshold:
                    skip_periods += 1
                    continue
        
        if current_num_bets == 0:
            skip_periods += 1
            continue
        
        bet_periods += 1
        
        # ========== 根据模式设置种子 ==========
        if seed_mode == "date":
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
            seed_val = fixed_seed_value + method_seed_offset
        elif seed_mode == "random":
            seed_val = random.randint(0, 1000000) + method_seed_offset
        else:
            seed_val = 42 + method_seed_offset + i
        
        random.seed(seed_val)
        np.random.seed(seed_val)
        
        # 每5期重新训练一次
        model_key = f"{method_name}_{i // retrain_interval}"
        
        if model_key not in trained_models:
            # 训练数据：测试期之前的 train_window 期
            train_data = draws[i - train_window:i]
            
            if method_name == "方法1":
                generator = Method1NewRule(train_data, sum_method=sum_predict_method)
                generator.normal_threshold = st.session_state.get('adv_normal_threshold', 50)
                generator.normal_count_layer1 = st.session_state.get('adv_normal_count', 5)
                generator.cold_count_layer1 = st.session_state.get('adv_cold_count', 1)
                generator.temp_normal = st.session_state.get('adv_normal_temp', 0.8)
                generator.temp_cold = st.session_state.get('adv_cold_temp', 1.2)
                generator.sum_tolerance = st.session_state.get('adv_sum_tolerance', 12)
                generator.require_consecutive = st.session_state.get('adv_require_consecutive', True)
                generator.max_attempts_layer1 = st.session_state.get('adv_max_attempts', 500)
                generator.max_attempts_layer2 = st.session_state.get('adv_max_attempts', 500) // 2
                
                generator.enable_freq_acc = st.session_state.get('enable_freq_acc', True)
                generator.enable_density_trend = st.session_state.get('enable_density_trend', True)
                generator.enable_absence_bonus = st.session_state.get('enable_absence_bonus', True)
                generator.enable_alternating = st.session_state.get('enable_alternating', True)

                # ========== 调试输出 ==========
                print(f"=== 回测 {method_name} 加分项状态 ===")
                print(f"generator.enable_freq_acc = {generator.enable_freq_acc}")
                print(f"generator.enable_density_trend = {generator.enable_density_trend}")
                print(f"generator.enable_absence_bonus = {generator.enable_absence_bonus}")
                print(f"generator.enable_alternating = {generator.enable_alternating}")
                print(f"st.session_state enable_freq_acc = {st.session_state.get('enable_freq_acc', 'NOT SET')}")
                # ================================
                
                bets = generator.generate_bets(current_num_bets, "7+1")
                
                blue_system = BlueScoreSystem(train_data, blue_method=blue_predict_method)
                for bet in bets:
                    blue = blue_system.select_blue()
                    bet['blues'] = [blue]
                    bet['blue'] = blue
                trained_models[model_key] = bets
            
            elif method_name == "方法2":
                generator = Method2DanTuo(train_data)
                bets = generator.generate_bets(current_num_bets, "7+1")
                blue_system = BlueScoreSystem(train_data, blue_method=blue_predict_method)
                for bet in bets:
                    blue = blue_system.select_blue()
                    bet['blues'] = [blue]
                    bet['blue'] = blue
                trained_models[model_key] = bets
            
            elif method_name == "方法3":
                generator = Method3LightGBM(train_data)
                generator.train()
                bets = generator.generate_bets(current_num_bets, "7+1")
                trained_models[model_key] = bets
            
            elif method_name == "方法4":
                generator = Method4Ensemble(train_data)
                generator.train()
                bets = generator.generate_bets(current_num_bets, "7+1")
                trained_models[model_key] = bets
            
            elif method_name == "方法5":
                methods_results = {}
                for m in ["方法1", "方法2", "方法3", "方法4"]:
                    m_key = f"{m}_{i // retrain_interval}"
                    if m_key not in trained_models:
                        if m == "方法1":
                            g = Method1NewRule(train_data, sum_method=sum_predict_method)
                            g.enable_freq_acc = st.session_state.get('enable_freq_acc', True)
                            g.enable_density_trend = st.session_state.get('enable_density_trend', True)
                            g.enable_absence_bonus = st.session_state.get('enable_absence_bonus', True)
                            g.enable_alternating = st.session_state.get('enable_alternating', True)
                            b = g.generate_bets(current_num_bets, "7+1")
                            bs = BlueScoreSystem(train_data, blue_method=blue_predict_method)
                            for bet in b:
                                blue = bs.select_blue()
                                bet['blues'] = [blue]
                                bet['blue'] = blue
                            trained_models[m_key] = b
                        elif m == "方法2":
                            g = Method2DanTuo(train_data)
                            b = g.generate_bets(current_num_bets, "7+1")
                            bs = BlueScoreSystem(train_data, blue_method=blue_predict_method)
                            for bet in b:
                                blue = bs.select_blue()
                                bet['blues'] = [blue]
                                bet['blue'] = blue
                            trained_models[m_key] = b
                        elif m == "方法3":
                            g = Method3LightGBM(train_data)
                            g.train()
                            b = g.generate_bets(current_num_bets, "7+1")
                            trained_models[m_key] = b
                        elif m == "方法4":
                            g = Method4Ensemble(train_data)
                            g.train()
                            b = g.generate_bets(current_num_bets, "7+1")
                            trained_models[m_key] = b
                    methods_results[m] = trained_models[m_key]
                
                all_reds = []
                all_blues = []
                for m in ["方法1", "方法2", "方法3", "方法4"]:
                    for bet in methods_results[m]:
                        all_reds.extend(bet['reds'])
                        all_blues.extend(bet.get('blues', [bet['blue']]))
                
                from collections import Counter
                red_counter = Counter(all_reds)
                blue_counter = Counter(all_blues)
                
                top_reds = [num for num, _ in red_counter.most_common(7)]
                top_reds.sort()
                top_blues = [num for num, _ in blue_counter.most_common(2)]
                
                bets = [{
                    'reds': top_reds,
                    'blues': top_blues,
                    'blue': top_blues[0] if top_blues else 8,
                    'sum': sum(top_reds),
                    'method': '方法5:综合模式'
                }]
                trained_models[model_key] = bets
            
            else:
                generator = Method1NewRule(train_data, sum_method=sum_predict_method)
                bets = generator.generate_bets(current_num_bets, "7+1")
                blue_system = BlueScoreSystem(train_data, blue_method=blue_predict_method)
                for bet in bets:
                    blue = blue_system.select_blue()
                    bet['blues'] = [blue]
                    bet['blue'] = blue
                trained_models[model_key] = bets
        else:
            bets = trained_models[model_key]
        
        # 计算中奖
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
        
        total_cost += current_num_bets * 14
        total_prize += period_prize
        
        # ========== 新增：记录中奖明细 ==========
        if period_prize > 0:
            win_count += 1
            prize_details.append({
                "period": test_period,
                "prize": period_prize
            })
    
    periods = lookback
    net = total_prize - total_cost
    roi = (net / total_cost) * 100 if total_cost > 0 else 0
    win_rate = (win_count / bet_periods) * 100 if bet_periods > 0 else 0
    
    # 格式化中奖明细字符串
    prize_details_str = ""
    if prize_details:
        detail_list = [f"{d['period']}({d['prize']}元)" for d in prize_details]
        prize_details_str = ", ".join(detail_list)
    
    return {
        "roi": roi,
        "total_cost": total_cost,
        "total_prize": total_prize,
        "net": net,
        "win_rate": win_rate,
        "periods": periods,
        "bet_periods": bet_periods,
        "skip_periods": skip_periods,
        "train_window_used": train_window,
        "prize_breakdown": prize_breakdown,
        "prize_details": prize_details_str,  # 新增
        "prize_details_list": prize_details   # 新增（备用）
    }
    
print("第4部分加载完成（v14.1 - 支持3种种子模式）")
print("=" * 60)
print("请确认第4部分代码，输入 CONFIRM 后继续第5部分")
print("=" * 60)

# ============================================================
# 第5部分：主页面UI + 动态投注 + 回测面板 + 侧边栏
# 版本：v15.0
# 修改内容：
#   1. 方法1替换为新规则系统 v15.0
#   2. 删除"连号/跳号"和"上期重复1-2个"复选框
#   3. 增加高级参数折叠面板（方法1专用）
#   4. 蓝球统一使用 BlueScoreSystem
#   5. 回测函数调用新方法1
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

# ========== 预测方法默认值 ==========
if 'sum_predict_method' not in st.session_state:
    st.session_state['sum_predict_method'] = '7期均值'
if 'blue_predict_method' not in st.session_state:
    st.session_state['blue_predict_method'] = '7期均值'

# ========== 基础分自定义默认值（50分 → 赋以概率 6.0%） ==========
if 'score_0' not in st.session_state:
    st.session_state['score_0'] = 70
if 'score_1_5' not in st.session_state:
    st.session_state['score_1_5'] = 50
if 'score_6_10' not in st.session_state:
    st.session_state['score_6_10'] = 20
if 'score_11_15' not in st.session_state:
    st.session_state['score_11_15'] = 15
if 'score_16_20' not in st.session_state:
    st.session_state['score_16_20'] = 10
if 'score_21_30' not in st.session_state:
    st.session_state['score_21_30'] = 8
if 'score_30_plus' not in st.session_state:
    st.session_state['score_30_plus'] = 5

# ========== 加分项分值默认值 ==========
if 'bonus_freq_acc' not in st.session_state:
    st.session_state['bonus_freq_acc'] = 25
if 'bonus_density_trend' not in st.session_state:
    st.session_state['bonus_density_trend'] = 20
if 'bonus_absence' not in st.session_state:
    st.session_state['bonus_absence'] = 30
if 'bonus_alternating' not in st.session_state:
    st.session_state['bonus_alternating'] = 12
# ==================== 主页面标题 ====================
col_title, col_settings = st.columns([0.9, 0.1])
with col_title:
    st.title("🎯 双色球AI智能选号工具 - 专业版 v15.0")
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
#---------
# ==================== 冷热码分析（新规则系统 v15.0） ====================
st.subheader("🔥 冷热码分析（基于综合评分）")

st.caption("📊 基于最近100期数据，使用综合评分体系（基础分+可选加分项）")

# 获取当前加分项开关状态（从高级参数面板读取）
enable_freq_acc = st.session_state.get('enable_freq_acc', True)
enable_density_trend = st.session_state.get('enable_density_trend', True)
enable_absence_bonus = st.session_state.get('enable_absence_bonus', True)
enable_alternating = st.session_state.get('enable_alternating', True)

# 显示当前评分规则状态
bonus_status = []
if enable_freq_acc:
    bonus_status.append("频率加速度")
if enable_density_trend:
    bonus_status.append("疏转密")
if enable_absence_bonus:
    bonus_status.append("遗漏13-20期")
if enable_alternating:
    bonus_status.append("隔期模式")

if bonus_status:
    st.caption(f"✅ 当前启用加分项: {' + '.join(bonus_status)}")
else:
    st.caption("⚠️ 当前仅使用基础分（遗漏值），未启用任何加分项")

# 创建评分实例并设置开关状态
scorer = Method1NewRule(draws)
scorer.enable_freq_acc = enable_freq_acc
scorer.enable_density_trend = enable_density_trend
scorer.enable_absence_bonus = enable_absence_bonus
scorer.enable_alternating = enable_alternating

# 重新计算评分（确保使用最新的开关状态）
scorer._calculate_scores_and_pools()
red_scores = scorer.red_scores

# 计算蓝球评分
blue_scores = get_blue_scores_by_new_system(draws)

# 红球评分排序
sorted_reds = sorted(red_scores.items(), key=lambda x: x[1], reverse=True)
hot_reds_top16 = sorted_reds[:16]      # 热门红球 Top 16
cold_reds_bottom16 = sorted_reds[-16:] # 冷门红球 Bottom 16

# 蓝球评分排序
sorted_blues = sorted(blue_scores.items(), key=lambda x: x[1], reverse=True)
hot_blues_top16 = sorted_blues[:16]    # 篮球热度 Top 16

# 辅助函数：生成居中HTML表格（放在冷热码分析之前，或者直接放在这里）
def make_centered_table(headers, rows):
    """生成居中对齐的HTML表格"""
    html = '<table style="width:100%; text-align:center; border-collapse:collapse;">'
    # 表头
    html += '<tr style="background-color:#f0f0f0;">'
    for h in headers:
        html += f'<th style="text-align:center; padding:8px;">{h}</th>'
    html += '<tr>'
    # 数据行
    for row in rows:
        html += '<tr>'
        for cell in row:
            html += f'<td style="text-align:center; padding:6px;">{cell}</td>'
        html += '</tr>'
    html += '</table>'
    return html


# 并排显示3列
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔥 热门红球 Top 16**")
    hot_reds_rows = [[f"{num:02d}", score] for num, score in hot_reds_top16]
    hot_reds_html = make_centered_table(['号码', '评分'], hot_reds_rows)
    st.markdown(hot_reds_html, unsafe_allow_html=True)

with col2:
    st.markdown("**❄️ 冷门红球 Bottom 16**")
    cold_reds_rows = [[f"{num:02d}", score] for num, score in cold_reds_bottom16]
    cold_reds_html = make_centered_table(['号码', '评分'], cold_reds_rows)
    st.markdown(cold_reds_html, unsafe_allow_html=True)

with col3:
    st.markdown("**💙 篮球热度 Top 16**")
    hot_blues_rows = [[f"{num:02d}", score] for num, score in hot_blues_top16]
    hot_blues_html = make_centered_table(['蓝球', '评分'], hot_blues_rows)
    st.markdown(hot_blues_html, unsafe_allow_html=True)

st.markdown("---")
#------------
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

# 计算两种预测方法的范围
sine_lower, sine_upper = get_target_sum_sine_range(draws)
range_lower, range_upper = get_target_sum_range(draws)

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

st.markdown("**📊 和值预测参考**")

# 显示两种预测方法的结果（统一显示范围）
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("理论均值", f"{RED_EXPECTED_SUM}")
with col2:
    st.metric("历史均值", f"{sum_trend['mean_sum']:.1f}")
with col3:
    st.metric("正弦拟合", f"{sine_lower}-{sine_upper}", delta="±12")
with col4:
    st.metric("7期均值", f"{range_lower}-{range_upper}", delta="±12")
with col5:
    current_sum = sum(draws[-1].get('reds', []))
    st.metric("当前和值", f"{current_sum}", delta=f"{current_sum - RED_EXPECTED_SUM:+d}")

# 和值预测方法选择
st.markdown("**🎯 和值预测方法**")
col_method1, col_method2 = st.columns(2)
with col_method1:
    use_sine_sum = st.radio(
        "选择预测方法",
        options=["正弦拟合", "7期均值"],
        index=1,  # 默认7期均值
        key="sum_predict_method",
        horizontal=True
    )
with col_method2:
    if use_sine_sum == "正弦拟合":
        st.info(f"当前预测范围: {sine_lower}-{sine_upper}")
    else:
        st.info(f"当前预测范围: {range_lower}-{range_upper}")

st.markdown("---")
#------------------
# ==================== 蓝球走势分析 ====================
st.subheader("💙 蓝球走势分析")

# 计算两种预测方法的结果
blue_sine_prediction, recent_blues, recent_periods = get_blue_sine_prediction(draws, window=8)
blue_sine_candidates = get_blue_sine_candidates(draws, window=8, radius=3)
blue_range_candidates = get_blue_range(draws, window=7, radius=3)

st.caption(f"📊 正弦拟合基于最近8期 | 7期均值基于最近7期（环形±3，各7个候选）")

# 获取50期蓝球数据
blue_series_50, period_series_50 = get_blue_series_for_plot(draws, lookback=50)

if len(blue_series_50) >= 10:
    try:
        # 创建图表
        fig_blue = go.Figure()
        
        # 绘制实际蓝球（50期，用点表示）
        fig_blue.add_trace(go.Scatter(
            x=list(range(len(blue_series_50))),
            y=blue_series_50,
            mode='markers',
            name='实际蓝球',
            marker=dict(color='#1f77b4', size=8, symbol='circle')
        ))
        
        # 绘制正弦拟合预测线（基于最近8期）
        if len(recent_blues) >= 6:
            fit_x = list(range(len(blue_series_50) - len(recent_blues), len(blue_series_50)))
            fit_y = recent_blues
            
            fig_blue.add_trace(go.Scatter(
                x=fit_x,
                y=fit_y,
                mode='lines+markers',
                name='正弦拟合（最近8期）',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(color='#ff7f0e', size=6)
            ))
            
            # 标记正弦拟合预测点
            fig_blue.add_trace(go.Scatter(
                x=[len(blue_series_50)],
                y=[blue_sine_prediction],
                mode='markers',
                name=f'正弦拟合预测点: {blue_sine_prediction:02d}',
                marker=dict(color='red', size=12, symbol='star', line=dict(width=2, color='darkred'))
            ))
        
        # 添加理论均值线（8.5）
        fig_blue.add_hline(
            y=8.5,
            line_dash="dash",
            line_color="green",
            annotation_text="理论均值(8.5)",
            annotation_position="top right"
        )
        
        # 设置Y轴范围
        fig_blue.update_yaxes(range=[0.5, 16.5], tickmode='linear', tick0=1, dtick=1)
        
        # 设置布局
        fig_blue.update_layout(
            title="最近50期蓝球走势",
            xaxis_title="期数（倒序）",
            yaxis_title="蓝球号码",
            height=450,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_blue, use_container_width=True)
        
        # 显示预测信息
        st.markdown("**📊 蓝球预测参考**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sine_candidates_str = ' '.join([f"{c:02d}" for c in blue_sine_candidates])
            st.metric("正弦拟合候选", f"{len(blue_sine_candidates)}个", delta=sine_candidates_str)
        with col2:
            range_candidates_str = ' '.join([f"{c:02d}" for c in blue_range_candidates])
            st.metric("7期均值候选", f"{len(blue_range_candidates)}个", delta=range_candidates_str)
        with col3:
            last_blue = blue_series_50[-1] if blue_series_50 else 0
            st.metric("上期蓝球", f"{last_blue:02d}")
        
        # 蓝球预测方法选择
        st.markdown("**🎯 蓝球预测方法**")
        use_sine_blue = st.radio(
            "选择预测方法",
            options=["正弦拟合", "7期均值"],
            index=1,  # 默认7期均值
            key="blue_predict_method",
            horizontal=True
        )
        
        if use_sine_blue == "正弦拟合":
            st.info(f"当前候选池: {blue_sine_candidates}")
        else:
            st.info(f"当前候选池: {blue_range_candidates}")
    
    except Exception as e:
        st.warning(f"蓝球走势图绘制失败: {e}")
else:
    st.info(f"数据不足（需要10期，当前{len(blue_series_50)}期），无法绘制蓝球走势图")

st.markdown("---")

#-------------
# ==================== ML智能分析 ====================
st.subheader("🧠 ML智能分析引擎")

# 修改：使用固定50期窗口的独立版ML引擎
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
    # 修改：方法1为默认推荐，调整顺序
    ai_model = st.selectbox(
        "AI模型",
        ["方法1: 新规则系统(v15.0) ⭐推荐", "方法2: 胆拖混合", "方法3: LightGBM", "方法4: XGBoost", "方法5: 综合模式"],
        key="ai_model"
    )

# ==================== 删除两个复选框 ====================
# 原"连号/跳号要求"和"上期重复1-2个要求"已删除
# 连号要求已移入高级参数面板

# ==================== 高级参数折叠面板（方法1专用） ====================
if "方法1" in ai_model:
    with st.expander("🔧 高级参数设置（方法1专用）", expanded=False):
        st.markdown("**红球参数**")
        
        col_adv1, col_adv2, col_adv3 = st.columns(3)
        with col_adv1:
            normal_threshold = st.number_input("正常池阈值", min_value=30, max_value=70, value=50, step=5, key="adv_normal_threshold")
            normal_count = st.number_input("正常池抽取数（第1层）", min_value=3, max_value=6, value=5, step=1, key="adv_normal_count")
            cold_count = st.number_input("冷码池抽取数（第1层）", min_value=0, max_value=3, value=1, step=1, key="adv_cold_count")
        
        with col_adv2:
            normal_temp = st.slider("正常池温度", min_value=0.5, max_value=1.5, value=0.8, step=0.1, key="adv_normal_temp")
            cold_temp = st.slider("冷码池温度", min_value=0.8, max_value=2.0, value=1.2, step=0.1, key="adv_cold_temp")
            sum_tolerance = st.number_input("和值容差", min_value=8, max_value=20, value=12, step=1, key="adv_sum_tolerance")
        
        with col_adv3:
            require_consecutive = st.checkbox("要求连号（≥1个）", value=True, key="adv_require_consecutive")
            max_attempts = st.number_input("最大尝试次数", min_value=100, max_value=1000, value=500, step=100, key="adv_max_attempts")
        
        st.markdown("---")
        st.markdown("**🎚️ 基础分自定义（50分 → 赋以概率 6.0%）**")
        st.caption("调整每个遗漏区间的分数，分数越高被选中概率越大")
        
        col_score1, col_score2, col_score3 = st.columns(3)
        with col_score1:
            score_0 = st.number_input("0期（上期重号）", value=70, min_value=0, max_value=100, step=5, key="score_0")
            st.caption(f"→ 赋以概率: {score_0 * 0.12:.1f}%")
            score_1_5 = st.number_input("1-5期", value=50, min_value=0, max_value=100, step=5, key="score_1_5")
            st.caption(f"→ 赋以概率: {score_1_5 * 0.12:.1f}%")
            score_6_10 = st.number_input("6-10期", value=20, min_value=0, max_value=100, step=5, key="score_6_10")
            st.caption(f"→ 赋以概率: {score_6_10 * 0.12:.1f}%")
        with col_score2:
            score_11_15 = st.number_input("11-15期", value=15, min_value=0, max_value=100, step=5, key="score_11_15")
            st.caption(f"→ 赋以概率: {score_11_15 * 0.12:.1f}%")
            score_16_20 = st.number_input("16-20期", value=10, min_value=0, max_value=100, step=5, key="score_16_20")
            st.caption(f"→ 赋以概率: {score_16_20 * 0.12:.1f}%")
            score_21_30 = st.number_input("21-30期", value=8, min_value=0, max_value=100, step=5, key="score_21_30")
            st.caption(f"→ 赋以概率: {score_21_30 * 0.12:.1f}%")
        with col_score3:
            score_30_plus = st.number_input(">30期", value=5, min_value=0, max_value=100, step=5, key="score_30_plus")
            st.caption(f"→ 赋以概率: {score_30_plus * 0.12:.1f}%")
        
        st.markdown("---")
        st.markdown("**🎚️ 加分项分值自定义（分数越高，命中该条件的号码权重越大）**")
        
        col_bonus_adj1, col_bonus_adj2 = st.columns(2)
        with col_bonus_adj1:
            bonus_freq_acc = st.number_input("频率加速度 Δf", value=25, min_value=0, max_value=50, step=5, key="bonus_freq_acc")
            st.caption(f"→ 赋以概率增加: {bonus_freq_acc * 0.12:.1f}%")
            bonus_density_trend = st.number_input("疏转密 trend", value=20, min_value=0, max_value=50, step=5, key="bonus_density_trend")
            st.caption(f"→ 赋以概率增加: {bonus_density_trend * 0.12:.1f}%")
        with col_bonus_adj2:
            bonus_absence = st.number_input("遗漏13-20期", value=30, min_value=0, max_value=50, step=5, key="bonus_absence")
            st.caption(f"→ 赋以概率增加: {bonus_absence * 0.12:.1f}%")
            bonus_alternating = st.number_input("隔期模式", value=12, min_value=0, max_value=30, step=5, key="bonus_alternating")
            st.caption(f"→ 赋以概率增加: {bonus_alternating * 0.12:.1f}%")
        
        st.markdown("---")
        st.markdown("**加分项开关（基础分始终开启）**")
        
        col_bonus1, col_bonus2 = st.columns(2)
        with col_bonus1:
            enable_freq_acc = st.checkbox("频率加速度 Δf", value=True, key="enable_freq_acc")
            enable_density_trend = st.checkbox("疏转密 trend", value=True, key="enable_density_trend")
        with col_bonus2:
            enable_absence_bonus = st.checkbox("遗漏13-20期", value=True, key="enable_absence_bonus")
            enable_alternating = st.checkbox("隔期模式", value=True, key="enable_alternating")
        
        st.markdown("---")
        st.markdown("**蓝球参数**")
        
        col_adv4, col_adv5, col_adv6 = st.columns(3)
        with col_adv4:
            blue_temperature = st.slider("蓝球Softmax温度", min_value=0.5, max_value=1.5, value=0.8, step=0.1, key="adv_blue_temp")
            neighbor1_bonus = st.number_input("邻号(±1)加分", min_value=0, max_value=40, value=20, step=5, key="adv_neighbor1")
            neighbor2_bonus = st.number_input("邻号(±2)加分", min_value=0, max_value=20, value=5, step=5, key="adv_neighbor2")
        
        with col_adv5:
            sine_fit_bonus = st.number_input("正弦拟合加分", min_value=0, max_value=40, value=20, step=5, key="adv_sine_bonus")
            freq_acc_bonus = st.number_input("频率加速度加分", min_value=0, max_value=30, value=15, step=5, key="adv_freq_bonus")
            mid_absence_bonus = st.number_input("遗漏9-13期加分", min_value=0, max_value=30, value=12, step=3, key="adv_mid_bonus")
        
        with col_adv6:
            remainder_gap4_bonus = st.number_input("除3余数空缺4-5期加分", min_value=0, max_value=30, value=15, step=5, key="adv_remainder4")
            remainder_gap6_bonus = st.number_input("除3余数空缺6-7期加分", min_value=0, max_value=20, value=10, step=5, key="adv_remainder6")
else:
    # 未选择方法1时，设置默认值
    normal_threshold = 50
    normal_count = 5
    cold_count = 1
    normal_temp = 0.8
    cold_temp = 1.2
    sum_tolerance = 12
    require_consecutive = True
    max_attempts = 500
    # 加分项开关默认值
    enable_freq_acc = True
    enable_density_trend = True
    enable_absence_bonus = True
    enable_alternating = True
    # 蓝球参数默认值
    blue_temperature = 0.8
    neighbor1_bonus = 20
    neighbor2_bonus = 5
    sine_fit_bonus = 20
    freq_acc_bonus = 15
    mid_absence_bonus = 12
    remainder_gap4_bonus = 15
    remainder_gap6_bonus = 10

# ==================== 随机种子设置 ====================
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

# ==================== 生成投注按钮 ====================
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
            
            # 根据选择的模型调用对应方法
            if "方法1" in ai_model:
                generator = Method1NewRule(draws)
                
                # 应用红球参数
                generator.normal_threshold = normal_threshold
                generator.normal_count_layer1 = normal_count
                generator.cold_count_layer1 = cold_count
                generator.temp_normal = normal_temp
                generator.temp_cold = cold_temp
                generator.sum_tolerance = sum_tolerance
                generator.require_consecutive = require_consecutive
                generator.max_attempts_layer1 = max_attempts
                generator.max_attempts_layer2 = max_attempts // 2
                
                # ========== 应用加分项开关 ==========
                generator.enable_freq_acc = enable_freq_acc
                generator.enable_density_trend = enable_density_trend
                generator.enable_absence_bonus = enable_absence_bonus
                generator.enable_alternating = enable_alternating
                
                bets = generator.generate_bets(num_bets, bet_type_code)
                
                # 使用新蓝球系统
                blue_system = BlueScoreSystem(draws)
                blue_system.temperature = blue_temperature
                blue_system.neighbor1_bonus = neighbor1_bonus
                blue_system.neighbor2_bonus = neighbor2_bonus
                blue_system.sine_fit_bonus = sine_fit_bonus
                blue_system.freq_acc_bonus = freq_acc_bonus
                blue_system.mid_absence_bonus = mid_absence_bonus
                blue_system.remainder_gap4_bonus = remainder_gap4_bonus
                blue_system.remainder_gap6_bonus = remainder_gap6_bonus
                
                for bet in bets:
                    blue = blue_system.select_blue()
                    bet['blues'] = [blue]
                    bet['blue'] = blue
            
            elif "方法2" in ai_model:
                generator = Method2DanTuo(draws)
                bets = generator.generate_bets(num_bets, bet_type_code)
                # 使用新蓝球系统
                blue_system = BlueScoreSystem(draws)
                blue_system.temperature = blue_temperature
                for bet in bets:
                    blue = blue_system.select_blue()
                    bet['blues'] = [blue]
                    bet['blue'] = blue
            
            elif "方法3" in ai_model:
                generator = Method3LightGBM(draws)
                bets = generator.generate_bets(num_bets, bet_type_code)
                # 方法3内部已有蓝球逻辑
            
            elif "方法4" in ai_model:
                generator = Method4Ensemble(draws)
                bets = generator.generate_bets(num_bets, bet_type_code)
                # 方法4内部已有蓝球逻辑
            
            else:  # 方法5: 综合模式
                # 综合模式：基于新方法1-4投票
                # 方法1投票
                generator1 = Method1NewRule(draws)
                bets1 = generator1.generate_bets(num_bets, bet_type_code)
                # 方法2投票
                generator2 = Method2DanTuo(draws)
                bets2 = generator2.generate_bets(num_bets, bet_type_code)
                # 方法3投票
                generator3 = Method3LightGBM(draws)
                bets3 = generator3.generate_bets(num_bets, bet_type_code)
                # 方法4投票
                generator4 = Method4Ensemble(draws)
                bets4 = generator4.generate_bets(num_bets, bet_type_code)
                
                # 合并所有投注，统计频率
                all_reds = []
                all_blues = []
                for bet in bets1 + bets2 + bets3 + bets4:
                    all_reds.extend(bet['reds'])
                    all_blues.extend(bet.get('blues', [bet['blue']]))
                
                from collections import Counter
                red_counter = Counter(all_reds)
                blue_counter = Counter(all_blues)
                
                # 取频率最高的7红+2蓝
                top_reds = [num for num, _ in red_counter.most_common(7)]
                top_reds.sort()
                top_blues = [num for num, _ in blue_counter.most_common(2)]
                
                bets = [{
                    'reds': top_reds,
                    'blues': top_blues,
                    'blue': top_blues[0],
                    'sum': sum(top_reds),
                    'method': '方法5:综合模式(v15.0)'
                }]
            
            st.session_state['generated_bets'] = bets
            st.session_state['model_used'] = ai_model
            st.session_state['last_bet_type'] = bet_type_code
        
        st.success(f"✅ 使用 {ai_model} 生成 {len(bets)} 组 {bet_type_code} 复式投注")

# ==================== 显示投注结果 ====================
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
#-------------
# ==================== ROI回测分析 ====================
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
            "每期基础组数",
            min_value=1,
            max_value=10,
            value=4,
            key="backtest_bets"
        )
    
    # 种子模式选择
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
    #----------------
    # ========== 新增：回测时使用的预测方法选择 ==========
    st.markdown("**🎯 回测时使用的预测方法**")
    col_test1, col_test2 = st.columns(2)
    with col_test1:
        test_sum_method = st.selectbox(
            "和值预测方法",
            options=["正弦拟合", "7期均值"],
            index=1,  # 默认7期均值
            key="test_sum_method"
        )
    with col_test2:
        test_blue_method = st.selectbox(
            "蓝球预测方法",
            options=["正弦拟合", "7期均值"],
            index=1,  # 默认7期均值
            key="test_blue_method"
        )
    # ================================================
    
    # ========== ML信号过滤 ==========
    st.markdown("**🧠 ML信号过滤**")
    col_signal1, col_signal2 = st.columns(2)
    with col_signal1:
        use_ml_signal = st.checkbox(
            "跟随ML智能分析投注",
            value=False,
            key="use_ml_signal",
            help="仅当信号强度≥阈值时才投注，低于阈值则不买"
        )
    with col_signal2:
        signal_threshold = st.slider(
            "信号强度阈值",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            key="signal_threshold",
            disabled=not use_ml_signal,
            help="信号强度≥此值时投注，否则观望"
        )
    
    # 显示信号强度说明
    if use_ml_signal:
        st.caption("💡 信号强度说明：≥60强烈推荐 | 40-59谨慎 | 20-39观望 | <20不买")
    
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
            # ========== 在这里添加调试代码 ==========
            st.write("当前加分项开关状态:")
            st.write(f"enable_freq_acc: {st.session_state.get('enable_freq_acc', 'NOT SET')}")
            st.write(f"enable_density_trend: {st.session_state.get('enable_density_trend', 'NOT SET')}")
            st.write(f"enable_absence_bonus: {st.session_state.get('enable_absence_bonus', 'NOT SET')}")
            st.write(f"enable_alternating: {st.session_state.get('enable_alternating', 'NOT SET')}")
            st.write(f"加分项分值:")
            st.write(f"bonus_freq_acc: {st.session_state.get('bonus_freq_acc', 'NOT SET')}")
            st.write(f"bonus_density_trend: {st.session_state.get('bonus_density_trend', 'NOT SET')}")
            st.write(f"bonus_absence: {st.session_state.get('bonus_absence', 'NOT SET')}")
            st.write(f"bonus_alternating: {st.session_state.get('bonus_alternating', 'NOT SET')}")
            # =====================================
            # 显示当前设置
            if seed_mode == "date":
                st.info("🔬 种子模式：每期使用自己的开奖日期+21:15")
            elif seed_mode == "fixed":
                st.info(f"🔬 种子模式：固定种子 = {fixed_seed_value}")
            else:
                st.info("🔬 种子模式：每期随机生成种子")
            
            if use_ml_signal:
                st.info(f"🧠 ML信号过滤：已启用，阈值={signal_threshold}%，信号不足时不投注")
            else:
                st.info("🧠 ML信号过滤：未启用，每期都投注")
            
            st.info(f"🎯 和值预测：{test_sum_method} | 蓝球预测：{test_blue_method}")
            
            with st.spinner(f"正在回测 {backtest_periods} 期，请稍候..."):
                results_data = []
                
                # 方法1：新规则系统
                result = backtest_roi(
                    draws, "方法1", backtest_bets, backtest_periods,
                    seed_mode=seed_mode, fixed_seed_value=fixed_seed_value,
                    use_ml_signal=use_ml_signal, signal_threshold=signal_threshold,
                    use_dynamic_bets=False,
                    sum_predict_method=test_sum_method,
                    blue_predict_method=test_blue_method
                )
                results_data.append({
                    "方法": "方法1:新规则系统(v15.0)",
                    "ROI": float(result.get('roi', 0)),
                    "总成本": int(result.get('total_cost', 0)),
                    "总奖金": int(result.get('total_prize', 0)),
                    "净收益": int(result.get('net', 0)),
                    "中奖率": float(result.get('win_rate', 0)),
                    "投注期数": result.get('bet_periods', 0),
                    "跳过期数": result.get('skip_periods', 0),
                    "中奖明细": result.get('prize_details', '')
                })
                
                # 方法2：胆拖混合
                result = backtest_roi(
                    draws, "方法2", backtest_bets, backtest_periods,
                    seed_mode=seed_mode, fixed_seed_value=fixed_seed_value,
                    use_ml_signal=use_ml_signal, signal_threshold=signal_threshold,
                    use_dynamic_bets=False,
                    sum_predict_method=test_sum_method,
                    blue_predict_method=test_blue_method
                )
                results_data.append({
                    "方法": "方法2:胆拖混合",
                    "ROI": float(result.get('roi', 0)),
                    "总成本": int(result.get('total_cost', 0)),
                    "总奖金": int(result.get('total_prize', 0)),
                    "净收益": int(result.get('net', 0)),
                    "中奖率": float(result.get('win_rate', 0)),
                    "投注期数": result.get('bet_periods', 0),
                    "跳过期数": result.get('skip_periods', 0),
                    "中奖明细": result.get('prize_details', '')
                })
                
                # 方法3：LightGBM
                result = backtest_roi(
                    draws, "方法3", backtest_bets, backtest_periods,
                    seed_mode=seed_mode, fixed_seed_value=fixed_seed_value,
                    use_ml_signal=use_ml_signal, signal_threshold=signal_threshold,
                    use_dynamic_bets=False,
                    sum_predict_method=test_sum_method,
                    blue_predict_method=test_blue_method
                )
                results_data.append({
                    "方法": "方法3:LightGBM",
                    "ROI": float(result.get('roi', 0)),
                    "总成本": int(result.get('total_cost', 0)),
                    "总奖金": int(result.get('total_prize', 0)),
                    "净收益": int(result.get('net', 0)),
                    "中奖率": float(result.get('win_rate', 0)),
                    "投注期数": result.get('bet_periods', 0),
                    "跳过期数": result.get('skip_periods', 0),
                    "中奖明细": result.get('prize_details', '')
                })
                
                # 方法4：XGBoost
                result = backtest_roi(
                    draws, "方法4", backtest_bets, backtest_periods,
                    seed_mode=seed_mode, fixed_seed_value=fixed_seed_value,
                    use_ml_signal=use_ml_signal, signal_threshold=signal_threshold,
                    use_dynamic_bets=False,
                    sum_predict_method=test_sum_method,
                    blue_predict_method=test_blue_method
                )
                results_data.append({
                    "方法": "方法4:XGBoost",
                    "ROI": float(result.get('roi', 0)),
                    "总成本": int(result.get('total_cost', 0)),
                    "总奖金": int(result.get('total_prize', 0)),
                    "净收益": int(result.get('net', 0)),
                    "中奖率": float(result.get('win_rate', 0)),
                    "投注期数": result.get('bet_periods', 0),
                    "跳过期数": result.get('skip_periods', 0),
                    "中奖明细": result.get('prize_details', '')
                })
                
                # 方法5：综合模式
                result = backtest_roi(
                    draws, "方法5", backtest_bets, backtest_periods,
                    seed_mode=seed_mode, fixed_seed_value=fixed_seed_value,
                    use_ml_signal=use_ml_signal, signal_threshold=signal_threshold,
                    use_dynamic_bets=False,
                    sum_predict_method=test_sum_method,
                    blue_predict_method=test_blue_method
                )
                results_data.append({
                    "方法": "方法5:综合模式",
                    "ROI": float(result.get('roi', 0)),
                    "总成本": int(result.get('total_cost', 0)),
                    "总奖金": int(result.get('total_prize', 0)),
                    "净收益": int(result.get('net', 0)),
                    "中奖率": float(result.get('win_rate', 0)),
                    "投注期数": result.get('bet_periods', 0),
                    "跳过期数": result.get('skip_periods', 0),
                    "中奖明细": result.get('prize_details', '')
                })
                
                df_results = pd.DataFrame(results_data)
                
                # 显示结果表格
                st.dataframe(
                    df_results,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        'ROI': st.column_config.NumberColumn('ROI', format='%.1f%%'),
                        '总成本': st.column_config.NumberColumn('总成本', format='¥%.0f'),
                        '总奖金': st.column_config.NumberColumn('总奖金', format='¥%.0f'),
                        '净收益': st.column_config.NumberColumn('净收益', format='¥%.0f'),
                        '中奖率': st.column_config.NumberColumn('中奖率', format='%.1f%%'),
                        '投注期数': st.column_config.NumberColumn('投注期数', format='%.0f'),
                        '跳过期数': st.column_config.NumberColumn('跳过期数', format='%.0f'),
                        '中奖明细': st.column_config.TextColumn('中奖明细', width='large')
                    }
                )
                
                # 找出最佳表现的方法（按ROI排序）
                best_method = results_data[0]["方法"]
                best_roi = results_data[0]["ROI"]
                for r in results_data:
                    if r["ROI"] > best_roi:
                        best_roi = r["ROI"]
                        best_method = r["方法"]
                
                st.success(f"🏆 最佳表现: {best_method} (ROI: {best_roi:.1f}%)")
                
                # 显示投注统计
                if use_ml_signal:
                    total_skip = results_data[0]["跳过期数"]
                    st.caption(f"📊 信号过滤：{backtest_periods}期中，{total_skip}期因信号不足未投注（节省成本约{total_skip * backtest_bets * 14}元）")
                
                st.caption(f"📅 基于最近{backtest_periods}期回测，每组{backtest_bets}注（7+1复式）")

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
    st.markdown("### 🎰 双色球AI分析工具 v15.0")
    st.markdown("---")
    
    with st.expander("🤖 ML库状态", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"{'✅' if LGB_AVAILABLE else '❌'} **LightGBM**")
            st.markdown(f"{'✅' if XGB_AVAILABLE else '❌'} **XGBoost**")
        with col2:
            st.markdown(f"{'✅' if SKLEARN_AVAILABLE else '❌'} **scikit-learn**")
        st.caption(f"MCP服务: {'✅ 可用' if MCP_AVAILABLE else '❌ 不可用'}")
    
    with st.expander("📖 算法说明 v15.0"):
        st.markdown("""
        | 算法 | 核心原理 | 特点 |
        |------|---------|------|
        | 🌟 方法1 | 新规则系统 | 评分+分层抽样+正弦拟合 ⭐推荐 |
        | 🟡 方法2 | 胆拖混合（新版） | Top5胆码+Softmax |
        | 🔵 方法3 | LightGBM | 梯度提升树 |
        | 🟣 方法4 | XGBoost | 极端梯度提升 |
        | 🌟 方法5 | 综合模式 | 新方法1-4投票 |
        
        **新规则系统 v15.0 核心特性**：
        - 单号码多维评分（基础分+4类加分）
        - 分池抽取（正常池+冷码池）
        - 动态降级策略（5+1 → 4+2）
        - 正弦拟合和值预测（±12）
        - 蓝球多维评分（8类加分）
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
    st.caption("DFSS智能选号工具 v15.0")
    st.caption("更新: 2026-05-25")
    st.caption("核心特性: 评分体系 | 分层抽样 | 正弦拟合 | 蓝球多维评分")


print("第5部分加载完成（v15.0 - UI修改版）")
print("=" * 60)
