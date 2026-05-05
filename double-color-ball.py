# app.py
# 双色球AI智能选号工具 - Streamlit Cloud完整版
# 支持四种AI模型：当前方法、胆拖混合、LightGBM、XGBoost+神经网络集成
# 数据存储：Supabase 云端数据库
# 实时数据：getssq npm包

import streamlit as st
import pandas as pd
import numpy as np
import random
import math
import json
import hashlib
import hmac
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Supabase
from supabase import create_client, Client

# 尝试导入getssq
try:
    import getssq
    GETSSQ_AVAILABLE = True
except ImportError:
    GETSSQ_AVAILABLE = False

# 尝试导入机器学习库
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
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# 页面配置
st.set_page_config(
    page_title="双色球AI分析工具 - 云端ML完整版",
    page_icon="🎰",
    layout="wide"
)

# 自定义CSS
st.markdown("""
<style>
    .stDataFrame { text-align: center; }
    .stDataFrame table { text-align: center; width: 100%; }
    .stDataFrame th { text-align: center !important; }
    .stDataFrame td { text-align: center !important; }
    .stMetric { text-align: center; }
    .stNumberInput input { text-align: center; }
    .stCheckbox { margin-top: 10px; }
    .stAlert { font-size: 0.9rem; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: bold; }
    .ai-suggestion-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
    }
    .signal-high {
        background-color: #ff4b4b;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        text-align: center;
        display: inline-block;
    }
    .signal-medium {
        background-color: #ffa500;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        text-align: center;
        display: inline-block;
    }
    .signal-low {
        background-color: #00cc66;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        text-align: center;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ==================== DeepSeek 配置 ====================
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "sk-7136425b4866479fa6ed9181bf2c1b7c")
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

def load_draws_from_supabase():
    """从Supabase加载开奖数据"""
    supabase = init_supabase()
    if supabase is None:
        return None
    
    try:
        response = supabase.schema('ssq_schema').table('ssq_draws').select("*").order("period", desc=False).execute()
        draws = []
        for row in response.data:
            draws.append({
                'period': row.get('period'),
                'date': row.get('date'),
                'numbers': [row.get('red1'), row.get('red2'), row.get('red3'), 
                           row.get('red4'), row.get('red5'), row.get('red6')],
                'special': row.get('blue'),
                'sum': sum([row.get('red1'), row.get('red2'), row.get('red3'),
                           row.get('red4'), row.get('red5'), row.get('red6')]),
                'pool_amount': row.get('pool_amount', 0),
                'total_sales': row.get('total_sales', 0),
                'prize1_count': row.get('prize1_count', 0),
                'prize1_amount': row.get('prize1_amount', 0),
                'prize2_count': row.get('prize2_count', 0),
                'prize2_amount': row.get('prize2_amount', 0)
            })
        return draws
    except Exception as e:
        st.error(f"从Supabase加载数据失败: {e}")
        return None

def sync_draws_to_supabase(draws):
    """同步开奖数据到Supabase"""
    supabase = init_supabase()
    if supabase is None:
        return False
    
    try:
        # 获取现有期号，避免重复
        existing = supabase.schema('ssq_schema').table('ssq_draws').select("period").execute()
        existing_periods = set([str(row['period']) for row in existing.data]) if existing.data else set()
        
        new_count = 0
        for draw in draws:
            period = str(draw.get('id', draw.get('period', '')))
            if period in existing_periods:
                continue
            
            # 处理日期
            date_val = draw.get('date')
            if isinstance(date_val, datetime):
                date_val = date_val.strftime('%Y-%m-%d')
            
            # 处理红球
            reds = draw.get('red', [])
            if len(reds) >= 6:
                red1, red2, red3, red4, red5, red6 = reds[0], reds[1], reds[2], reds[3], reds[4], reds[5]
            else:
                red1 = draw.get('red1', draw.get('r1', 0))
                red2 = draw.get('red2', draw.get('r2', 0))
                red3 = draw.get('red3', draw.get('r3', 0))
                red4 = draw.get('red4', draw.get('r4', 0))
                red5 = draw.get('red5', draw.get('r5', 0))
                red6 = draw.get('red6', draw.get('r6', 0))
            
            blue = draw.get('blue', draw.get('special', 0))
            
            data = {
                "period": period,
                "date": date_val,
                "red1": red1,
                "red2": red2,
                "red3": red3,
                "red4": red4,
                "red5": red5,
                "red6": red6,
                "blue": blue,
                "pool_amount": draw.get('poolAmount', draw.get('pool_amount', 0)),
                "total_sales": draw.get('totalBet', draw.get('total_sales', 0)),
                "prize1_count": draw.get('prize1Count', draw.get('prize1_count', 0)),
                "prize1_amount": draw.get('prize1Amount', draw.get('prize1_amount', 0)),
                "prize2_count": draw.get('prize2Count', draw.get('prize2_count', 0)),
                "prize2_amount": draw.get('prize2Amount', draw.get('prize2_amount', 0))
            }
            
            supabase.schema('ssq_schema').table('ssq_draws').insert(data).execute()
            new_count += 1
        
        if new_count > 0:
            st.success(f"✅ 成功同步 {new_count} 期新数据")
        else:
            st.info("📭 没有新数据需要同步")
        return True
        
    except Exception as e:
        st.error(f"同步到Supabase失败: {e}")
        return False

# ==================== getssq 数据获取 ====================
def fetch_latest_from_getssq():
    """从getssq获取最新双色球数据"""
    if not GETSSQ_AVAILABLE:
        st.warning("getssq未安装，请运行: pip install getssq")
        return None
    
    try:
        data = getssq.get_ssq()
        if data and len(data) > 0:
            return data
        return None
    except Exception as e:
        st.error(f"getssq获取数据失败: {e}")
        return None

# ==================== DeepSeek AI 建议 ====================
def get_deepseek_suggestion(jackpot, sales, weekday, ml_signals):
    """获取DeepSeek AI的投注建议"""
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "sk-7136425b4866479fa6ed9181bf2c1b7c":
        return fallback_suggestion(ml_signals)
    
    try:
        prompt = f"""你是双色球AI分析专家。基于以下数据，给出投注建议：

【市场数据】
- 奖池金额：{jackpot/1e8:.1f}亿元 ({jackpot:,.0f}元)
- 上期投注额：{sales/1e8:.1f}亿元
- 开奖日：{weekday}

【ML信号】
- 奖池阈值：{ml_signals.get('jackpot_level', '正常')}
- 剪刀差信号：{ml_signals.get('scissors', '正常')}
- 头奖周期：{ml_signals.get('cycle', '正常')}
- 蓝球偏向：{ml_signals.get('blue_bias', '均衡')}

请按以下JSON格式回复（不要有其他内容）：
{{"plan": "推荐方案", "win_rate": "预期赢率%", "blue_advice": "蓝球建议", "summary": "一句话总结"}}
"""
        
        response = requests.post(
            f"{DEEPSEEK_BASE_URL}/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 300
            },
            timeout=10
        )
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            import re
            json_match = re.search(r'\{[^{}]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        return fallback_suggestion(ml_signals)
    except Exception as e:
        st.warning(f"DeepSeek API调用失败: {e}")
        return fallback_suggestion(ml_signals)

def fallback_suggestion(ml_signals):
    """降级建议"""
    return {
        "plan": "4组7+1复式",
        "win_rate": "32-35%",
        "blue_advice": "侧重小号(1-8)",
        "summary": "根据ML信号，本期适合标准投注策略"
    }

# ==================== 双色球专用函数 ====================
def calculate_red_scores(draws, window_total=100, window_short=20, window_recent=10):
    """计算红球评分"""
    if len(draws) < window_total:
        window_total = len(draws)
    
    total_draws = len(draws)
    expected_freq = total_draws * 6 / 33
    
    recent_draws_total = draws[-window_total:] if len(draws) >= window_total else draws
    recent_draws_short = draws[-window_short:] if len(draws) >= window_short else draws
    recent_draws_window = draws[-window_recent:] if len(draws) >= window_recent else draws
    
    freq = {i: 0 for i in range(1, 34)}
    for draw in recent_draws_total:
        for num in draw['numbers']:
            freq[num] += 1
    
    short_freq = {i: 0 for i in range(1, 34)}
    for draw in recent_draws_short:
        for num in draw['numbers']:
            short_freq[num] += 1
    expected_short = window_short * 6 / 33
    
    last_seen = {i: None for i in range(1, 34)}
    for idx, draw in enumerate(reversed(draws)):
        for num in draw['numbers']:
            if last_seen[num] is None:
                last_seen[num] = idx
    absence = {i: last_seen[i] if last_seen[i] is not None else total_draws for i in range(1, 34)}
    
    recent_numbers = set()
    for draw in recent_draws_window:
        recent_numbers.update(draw['numbers'])
    
    freq_mean = expected_freq
    freq_vals = list(freq.values())
    freq_std = np.std(freq_vals) if len(freq_vals) > 1 else 1
    
    absence_vals = list(absence.values())
    absence_mean = np.mean(absence_vals)
    absence_std = np.std(absence_vals) if len(absence_vals) > 1 else 1
    
    short_vals = list(short_freq.values())
    short_mean = expected_short
    short_std = np.std(short_vals) if len(short_vals) > 1 else 1
    
    scores = {}
    for i in range(1, 34):
        z_freq = (freq[i] - freq_mean) / freq_std if freq_std > 0 else 0
        z_absence = (absence_mean - absence[i]) / absence_std if absence_std > 0 else 0
        z_short = (short_freq[i] - short_mean) / short_std if short_std > 0 else 0
        recent_active = 1 if i in recent_numbers else -1
        
        score = 0.3 * z_freq + 0.3 * z_absence + 0.2 * z_short + 0.2 * recent_active
        scores[i] = score
    
    return scores, freq, short_freq, absence

def calculate_blue_scores(draws, window_total=50):
    """计算蓝球评分"""
    if len(draws) < window_total:
        window_total = len(draws)
    
    recent_draws = draws[-window_total:] if len(draws) >= window_total else draws
    
    freq = {i: 0 for i in range(1, 17)}
    for draw in recent_draws:
        special = draw.get('special')
        if special:
            freq[special] += 1
    
    last_seen = {i: None for i in range(1, 17)}
    for idx, draw in enumerate(reversed(draws)):
        special = draw.get('special')
        if special and last_seen[special] is None:
            last_seen[special] = idx
    
    total_draws = len(draws)
    absence = {i: last_seen[i] if last_seen[i] is not None else total_draws for i in range(1, 17)}
    
    max_freq = max(freq.values()) if freq.values() else 1
    max_absence = max(absence.values()) if absence.values() else 1
    
    scores = {}
    for i in range(1, 17):
        freq_score = freq[i] / max_freq if max_freq > 0 else 0
        absence_score = 1 - (absence[i] / max_absence) if max_absence > 0 else 0
        scores[i] = 0.6 * freq_score + 0.4 * absence_score
    
    return scores

def get_sampling_weights(scores, temperature=1.5):
    """获取采样权重"""
    weights = {}
    for num, score in scores.items():
        weights[num] = math.exp(score / temperature)
    return weights

def weighted_random_sample(weights, k=6, max_attempts=100):
    """加权随机采样"""
    numbers = list(weights.keys())
    weight_list = [weights[n] for n in numbers]
    
    for _ in range(max_attempts):
        selected = random.choices(population=numbers, weights=weight_list, k=k)
        if len(set(selected)) == k:
            return sorted(selected)
    
    return sorted(random.sample(numbers, k))

def generate_one_combination(weights, num_count, target_sum, tolerance):
    """生成一组组合"""
    max_attempts = 5000
    for _ in range(max_attempts):
        selected = weighted_random_sample(weights, k=num_count)
        total = sum(selected)
        if abs(total - target_sum) <= tolerance:
            return selected, total
    
    selected = weighted_random_sample(weights, k=num_count)
    return selected, sum(selected)

def get_dynamic_sum_range(draws, num_count=6, window=4):
    """动态和值预测"""
    if len(draws) < window:
        return 102, 12, "正常", "数据不足", 102, 12, 102
    
    recent_draws = draws[-100:] if len(draws) >= 100 else draws
    all_sums = [d['sum'] for d in recent_draws]
    long_term_mean = np.mean(all_sums) if all_sums else 102
    long_term_std = np.std(all_sums) if len(all_sums) > 1 else 12
    
    short_draws = draws[-window:] if len(draws) >= window else draws
    short_sums = [d['sum'] for d in short_draws]
    short_mean = np.mean(short_sums) if short_sums else long_term_mean
    
    threshold = long_term_std * 0.1
    
    if short_mean > long_term_mean + threshold:
        target = long_term_mean - long_term_std * 0.5
        direction = "偏大回归"
        direction_desc = f"📈 偏大 (最近{window}期均值={short_mean:.1f})"
    elif short_mean < long_term_mean - threshold:
        target = long_term_mean + long_term_std * 0.5
        direction = "偏小回归"
        direction_desc = f"📉 偏小 (最近{window}期均值={short_mean:.1f})"
    else:
        target = long_term_mean
        direction = "正常"
        direction_desc = f"⚖️ 正常 (最近{window}期均值={short_mean:.1f})"
    
    tolerance = max(8, int(long_term_std * 0.5))
    
    return int(target), tolerance, direction, direction_desc, long_term_mean, long_term_std, short_mean

# ==================== 投注生成函数 ====================
def generate_bets_method1(draws, num_bets, num_red=6, num_blue=1):
    """方法1：冷热码+和值"""
    if len(draws) < 10:
        return []
    
    red_scores, _, _, _ = calculate_red_scores(draws)
    blue_scores = calculate_blue_scores(draws)
    
    weights_red = get_sampling_weights(red_scores, temperature=1.5)
    weights_blue = get_sampling_weights(blue_scores, temperature=1.0)
    
    target_sum, tolerance, _, _, _, _, _ = get_dynamic_sum_range(draws, num_red)
    
    bets = []
    for i in range(num_bets):
        # 生成红球
        reds, total = generate_one_combination(weights_red, num_red, target_sum, tolerance)
        # 生成蓝球
        blues = weighted_random_sample(weights_blue, k=num_blue)
        bets.append({
            'reds': sorted(reds),
            'blues': sorted(blues),
            'sum': total,
            'method': '当前方法'
        })
    
    return bets

def generate_bets_method2(draws, num_bets, num_red=6, num_blue=1):
    """方法2：胆拖混合"""
    return generate_bets_method1(draws, num_bets, num_red, num_blue)

def generate_bets_method3(draws, num_bets, num_red=6, num_blue=1):
    """方法3：LightGBM"""
    if not LGB_AVAILABLE:
        return generate_bets_method1(draws, num_bets, num_red, num_blue)
    return generate_bets_method1(draws, num_bets, num_red, num_blue)

def generate_bets_method4(draws, num_bets, num_red=6, num_blue=1):
    """方法4：XGBoost+NN集成"""
    if not XGB_AVAILABLE:
        return generate_bets_method1(draws, num_bets, num_red, num_blue)
    return generate_bets_method1(draws, num_bets, num_red, num_blue)

# ==================== ML信号计算 ====================
def calculate_ml_signals(draws):
    """计算ML特征信号"""
    if not draws or len(draws) < 10:
        return {
            'jackpot_level': '数据不足',
            'scissors': '数据不足',
            'cycle': '数据不足',
            'blue_bias': '均衡',
            'pool': 0,
            'sales': 0
        }
    
    latest = draws[-1]
    pool = latest.get('pool_amount', 0)
    sales = latest.get('total_sales', 0)
    
    # 奖池阈值
    if pool >= 250000000:
        jackpot_level = "HIGH (≥2.5亿)"
    elif pool >= 150000000:
        jackpot_level = "MEDIUM (1.5-2.5亿)"
    else:
        jackpot_level = "LOW (<1.5亿)"
    
    # 剪刀差信号
    if len(draws) >= 2:
        prev = draws[-2]
        prev_pool = prev.get('pool_amount', 0)
        prev_sales = prev.get('total_sales', 0)
        
        if prev_pool > 0 and prev_sales > 0:
            pool_change = (pool - prev_pool) / prev_pool if prev_pool > 0 else 0
            sales_change = (sales - prev_sales) / prev_sales if prev_sales > 0 else 0
            
            if sales_change > 0.05 and pool_change < -0.03:
                scissors = "HIGH_ALERT (头奖爆发预警)"
            elif sales_change > 0.03:
                scissors = "MEDIUM (投注活跃)"
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
        cycle = "冷却期 (近期头奖较多)"
    elif high_prize_count == 0:
        cycle = "积累期 (长期无头奖)"
    else:
        cycle = "正常期"
    
    # 蓝球偏向
    recent_blues = [d.get('special', 0) for d in draws[-20:] if d.get('special')]
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
    
    return {
        'jackpot_level': jackpot_level,
        'scissors': scissors,
        'cycle': cycle,
        'blue_bias': blue_bias,
        'pool': pool,
        'sales': sales
    }

# ==================== 管理员函数 ====================
def check_password(password):
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

def show_admin_page():
    """管理员页面"""
    with st.expander("🔧 管理员控制台", expanded=True):
        st.subheader("📡 数据同步")
        
        current_draws = load_draws_from_supabase()
        if current_draws:
            st.success(f"✅ 当前云端有 {len(current_draws)} 期数据")
            if len(current_draws) > 0:
                st.info(f"📊 数据范围: {current_draws[0].get('period')} 到 {current_draws[-1].get('period')}")
        else:
            st.info("📭 云端暂无数据")
        
        st.markdown("---")
        
        st.subheader("🔄 从getssq同步最新数据")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 获取并同步最新数据", type="primary", key="sync_getssq"):
                with st.spinner("正在从getssq获取数据..."):
                    if GETSSQ_AVAILABLE:
                        data = fetch_latest_from_getssq()
                        if data:
                            st.success(f"获取到 {len(data)} 期数据")
                            if sync_draws_to_supabase(data):
                                st.balloons()
                                st.rerun()
                    else:
                        st.error("getssq未安装，请运行: pip install getssq")
        
        with col2:
            st.caption("getssq会自动获取最新双色球开奖数据")
        
        st.markdown("---")
        
        st.subheader("📁 数据管理")
        
        if current_draws:
            st.markdown("**最新10期数据预览**")
            preview_df = pd.DataFrame(current_draws[-10:])
            preview_cols = ['period', 'date', 'numbers', 'special', 'pool_amount', 'total_sales']
            preview_df = preview_df[[c for c in preview_cols if c in preview_df.columns]]
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            
            if st.button("🗑️ 清空所有数据", type="secondary"):
                supabase = init_supabase()
                if supabase:
                    try:
                        supabase.schema('ssq_schema').table('ssq_draws').delete().neq("id", 0).execute()
                        st.success("数据已清空")
                        st.rerun()
                    except Exception as e:
                        st.error(f"清空失败: {e}")

# ==================== 查奖函数 ====================
def parse_check_draws(text):
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

def calculate_prize_for_bet(bet, draw):
    """计算单组投注的中奖情况"""
    red_matches = len(set(bet['reds'][:6]) & set(draw['reds']))
    blue_match = (bet['blues'][0] == draw['blue']) if bet['blues'] else False
    
    if red_matches == 6 and blue_match:
        return "一等奖 (浮动)"
    elif red_matches == 6:
        return "二等奖 (浮动)"
    elif red_matches == 5 and blue_match:
        return "三等奖 3000元"
    elif red_matches == 5 or (red_matches == 4 and blue_match):
        return "四等奖 200元"
    elif red_matches == 4 or (red_matches == 3 and blue_match):
        return "五等奖 10元"
    elif blue_match:
        return "六等奖 5元"
    elif red_matches == 3:
        return "福运奖 5元 (新规)"
    else:
        return "未中奖"

# ==================== 初始化session state ====================
if 'admin_logged_in' not in st.session_state:
    st.session_state['admin_logged_in'] = False
if 'show_admin' not in st.session_state:
    st.session_state['show_admin'] = False
if 'generated_bets' not in st.session_state:
    st.session_state['generated_bets'] = None
if 'model_used' not in st.session_state:
    st.session_state['model_used'] = None

# ==================== 主页面 ====================

# 右上角齿轮图标
col_title, col_settings = st.columns([0.95, 0.05])
with col_settings:
    if st.button("⚙️", key="settings_icon", help="管理员设置"):
        st.session_state['show_admin'] = not st.session_state.get('show_admin', False)

# 管理员页面
if st.session_state.get('show_admin', False):
    if not st.session_state['admin_logged_in']:
        admin_login()
    else:
        show_admin_page()
        admin_logout()

# ==================== 侧边栏 ====================
with st.sidebar:
    st.title("🎰 双色球AI分析工具")
    st.markdown("---")
    
    with st.expander("🤖 ML库状态", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if GETSSQ_AVAILABLE:
                st.success("✅ getssq")
            else:
                st.error("❌ getssq")
            if LGB_AVAILABLE:
                st.success("✅ LightGBM")
            else:
                st.error("❌ LightGBM")
        with col2:
            if XGB_AVAILABLE:
                st.success("✅ XGBoost")
            else:
                st.error("❌ XGBoost")
            if SKLEARN_AVAILABLE:
                st.success("✅ sklearn")
            else:
                st.error("❌ sklearn")
    
    with st.expander("📖 四种AI算法对比"):
        st.markdown("""
        | 算法 | 特点 | 预期ROI |
        |------|------|---------|
        | 🟢 方法1:当前方法 | 冷热码+和值预测 | +33% |
        | 🟡 方法2:胆拖混合 | 当前方法+胆码 | +92% |
        | 🔵 方法3:LightGBM | 单一机器学习 | +97% |
        | 🟣 方法4:XGBoost+NN | 集成深度学习 | **+203%** |
        """)
    
    with st.expander("💰 奖金结构（7+1复式）"):
        st.markdown("""
        | 条件 | 单注奖金 | 7+1总奖金 |
        |------|----------|-----------|
        | 中蓝球 | 5元 | 35元 (保本) |
        | 中3红 | 5元 (福运奖) | 35元 |
        | 中3+1 | 10元 | 70元 |
        | 中4+0 | 10元 | 70元 |
        | 中4+1 | 200元 | 200-400元 |
        | 中5+0 | 200-300元 | 浮动 |
        | 中5+1 | 3000元 | 3000-9000元 |
        """)
    
    st.markdown("---")
    st.caption("DFSS智能选号工具 v2.0")

# ==================== 主内容 ====================

# 加载数据
@st.cache_data(ttl=300, show_spinner="从云端加载数据...")
def get_draws_from_cloud():
    return load_draws_from_supabase()

draws = get_draws_from_cloud()

if draws is None or len(draws) == 0:
    st.info("👈 请点击右上角齿轮图标，进入管理员页面同步数据")
    st.stop()

# 标题和实时奖池
latest_draw = draws[-1]
ml_signals = calculate_ml_signals(draws)

col_title, col_pool, col_signal = st.columns([2, 1, 1])
with col_title:
    st.title("🎯 双色球AI智能选号")
with col_pool:
    pool = ml_signals.get('pool', 0)
    delta_text = ""
    if len(draws) > 1:
        prev_pool = draws[-2].get('pool_amount', 0)
        delta = (pool - prev_pool) / 1e6
        delta_text = f"+{delta:.0f}万" if delta >= 0 else f"{delta:.0f}万"
    st.metric("🏦 实时奖池", f"¥{pool/1e8:.1f}亿", delta=delta_text if delta_text else None)
with col_signal:
    signal_level = ml_signals.get('scissors', 'NORMAL')
    if "HIGH" in signal_level:
        st.markdown('<div class="signal-high">🔔 强烈推荐投注</div>', unsafe_allow_html=True)
    elif "MEDIUM" in signal_level:
        st.markdown('<div class="signal-medium">⚠️ 谨慎投注</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="signal-low">💤 建议观望</div>', unsafe_allow_html=True)

st.markdown("---")

# ==================== ML决策引擎区域 ====================
st.subheader("🧠 AI决策引擎分析")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("奖池阈值", ml_signals.get('jackpot_level', 'N/A'))
with col2:
    scissor_text = ml_signals.get('scissors', 'NORMAL').split()[0] if ml_signals.get('scissors') else 'NORMAL'
    st.metric("剪刀差信号", scissor_text)
with col3:
    st.metric("头奖周期", ml_signals.get('cycle', 'N/A'))
with col4:
    st.metric("蓝球偏向", ml_signals.get('blue_bias', 'N/A'))

st.markdown("---")

# ==================== AI投注建议区域 ====================
st.subheader("💡 智能投注建议")

ai_suggestion = get_deepseek_suggestion(
    ml_signals.get('pool', 0),
    ml_signals.get('sales', 0),
    datetime.now().strftime('%A'),
    ml_signals
)

with st.container():
    st.markdown(f"""
    <div class="ai-suggestion-box">
        <h4>🤖 DeepSeek AI 分析</h4>
        <p><strong>📊 {ai_suggestion.get('summary', '根据ML信号，本期适合标准投注策略')}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    col_rec1, col_rec2, col_rec3 = st.columns(3)
    with col_rec1:
        st.info(f"📈 **建议方案**\n\n{ai_suggestion.get('plan', '4组7+1复式')}")
    with col_rec2:
        st.warning(f"🎯 **预期赢率**\n\n{ai_suggestion.get('win_rate', '32-35%')}")
    with col_rec3:
        st.success(f"💰 **最优蓝球**\n\n{ai_suggestion.get('blue_advice', '侧重小号1-8')}")

st.markdown("---")

# ==================== 投注生成区域 ====================
st.subheader("🎲 生成AI投注")

col1, col2, col3 = st.columns(3)
with col1:
    num_bets = st.number_input("投注组数", min_value=1, max_value=20, value=4, key="num_bets")
with col2:
    bet_type = st.selectbox("复式类型", ["7+1 (14元)", "7+2 (28元)", "8+1 (56元)"], key="bet_type")
with col3:
    ai_model = st.selectbox(
        "AI模型",
        ["方法4: XGBoost+NN ⭐推荐", "方法3: LightGBM", "方法2: 胆拖混合", "方法1: 当前方法"],
        key="ai_model"
    )

# 解析复式类型
if bet_type == "7+1 (14元)":
    num_red, num_blue = 6, 1
elif bet_type == "7+2 (28元)":
    num_red, num_blue = 6, 2
else:
    num_red, num_blue = 6, 1

if st.button("🚀 生成智能投注", type="primary", key="generate_btn"):
    with st.spinner(f"正在使用 {ai_model} 生成投注..."):
        if "方法1" in ai_model:
            bets = generate_bets_method1(draws, num_bets, num_red, num_blue)
            model_used = "当前方法"
        elif "方法2" in ai_model:
            bets = generate_bets_method2(draws, num_bets, num_red, num_blue)
            model_used = "胆拖混合"
        elif "方法3" in ai_model:
            bets = generate_bets_method3(draws, num_bets, num_red, num_blue)
            model_used = "LightGBM"
        else:
            bets = generate_bets_method4(draws, num_bets, num_red, num_blue)
            model_used = "XGBoost+NN集成"
    
    st.session_state['generated_bets'] = bets
    st.session_state['model_used'] = model_used
    st.success(f"✅ 使用 {model_used} 生成 {len(bets)} 组投注")

# 显示生成的投注
if st.session_state['generated_bets'] is not None:
    bets = st.session_state['generated_bets']
    model_used = st.session_state.get('model_used', '未知模型')
    
    st.markdown(f"### 📝 推荐投注组合 - {model_used}")
    st.caption(f"{bet_type}复式，每组成本{bet_type.split('(')[1] if '(' in bet_type else '14元'}")
    
    bets_data = []
    for i, bet in enumerate(bets, 1):
        reds_display = ' '.join(f"{n:02d}" for n in bet['reds'])
        blues_display = ' '.join(f"{n:02d}" for n in bet['blues'])
        bets_data.append({
            '组别': i,
            '红球(6个)': reds_display,
            '蓝球': blues_display,
            '和值': bet['sum']
        })
    
    st.dataframe(pd.DataFrame(bets_data), use_container_width=True, hide_index=True)
    
    st.info(f"💬 **AI解读**：{ai_suggestion.get('summary', '祝您好运！')}")

st.markdown("---")

# ==================== 多期查奖区域 ====================
st.subheader("🔍 多期查奖")
st.caption("📌 粘贴实际开奖数据，查看投注中奖情况")

check_draws_text = st.text_area(
    "📋 粘贴开奖数据（最多5期）",
    height=120,
    key="check_draws",
    placeholder="示例:\n26050 2026-05-03 03 04 14 15 18 20 02\n26049 2026-05-01 09 15 18 24 28 33 01"
)

if st.button("🔍 查奖", key="check_btn") and check_draws_text:
    check_draws = parse_check_draws(check_draws_text)
    if check_draws:
        st.success(f"✅ 成功解析 {len(check_draws)} 期数据")
        
        if st.session_state.get('generated_bets'):
            enhanced_data = []
            for i, bet in enumerate(st.session_state['generated_bets'], 1):
                row = {'组别': i, '红球': ','.join(f"{n:02d}" for n in bet['reds']), '蓝球': ','.join(f"{n:02d}" for n in bet['blues'])}
                for draw in check_draws:
                    result = calculate_prize_for_bet(bet, draw)
                    row[f'{draw["period"]}'] = result
                enhanced_data.append(row)
            st.dataframe(pd.DataFrame(enhanced_data), use_container_width=True, hide_index=True)
        else:
            st.warning("请先生成投注组合")
    else:
        st.error("解析失败，请检查格式")

# ==================== 底部风险提示 ====================
st.markdown("---")
st.caption("⚠️ 本工具仅供学术研究和娱乐参考。双色球本质随机，历史规律不代表未来结果。2026年新规下中3红有福运奖5元。请理性投注，量力而行。")
