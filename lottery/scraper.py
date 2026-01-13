from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, List, Set, Dict

import requests
from bs4 import BeautifulSoup

from .database import Draw

HISTORY_URL = "https://datachart.500.com/ssq/history/inc/history.php?start=0001&end=9999"
CWL_API = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
}


def _to_number(text: str) -> float | None:
    """将带千分位或中文单位的字符串转数值."""
    if not text:
        return None
    clean = text.replace(",", "").replace(" ", "")
    if clean in {"--", ""}:
        return None
    try:
        if clean.endswith("亿"):
            return float(clean[:-1]) * 1e8
        if clean.endswith("万"):
            return float(clean[:-1]) * 1e4
        return float(clean)
    except ValueError:
        return None


def _to_int(text: str) -> int | None:
    if not text:
        return None
    clean = re.sub(r"[^\d]", "", text)
    return int(clean) if clean else None


def fetch_history_html(url: str = HISTORY_URL) -> str:
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def parse_history_500(html: str) -> List[Draw]:
    soup = BeautifulSoup(html, "html.parser")
    # 新版页面使用 table#tablelist + tbody#tdata
    container = soup.find("tbody", id="tdata") or soup.find(id="tdata")
    if container is None:
        tablelist = soup.find("table", id="tablelist")
        if tablelist:
            container = tablelist.find("tbody")
    if container is None:
        raise ValueError("未找到历史数据容器 tdata/tablelist，数据源结构可能已变化")

    draws: List[Draw] = []
    for row in container.find_all("tr"):
        cols = [td.get_text(strip=True) for td in row.find_all("td")]
        # 期号 + 6红 + 1蓝 + 若干统计列 + 开奖日期（末列）
        if len(cols) < 8:
            continue

        issue = cols[0]
        date_str = cols[-1]
        try:
            draw_date = datetime.strptime(date_str, "%Y-%m-%d").date().isoformat()
        except ValueError:
            # 某些行可能是广告/空行
            continue

        try:
            reds = [int(x) for x in cols[1:7]]
            blue = int(cols[7])
        except (ValueError, IndexError):
            continue

        # 索引对应表头：8=星期/期次，9=奖池，10/11=一等奖注数/奖金，12/13=二等奖注数/奖金，14=总投注额，15=开奖日期
        pool = _to_number(cols[9]) if len(cols) > 9 else None
        first_prize_count = _to_int(cols[10]) if len(cols) > 10 else None
        first_prize_amount = _to_number(cols[11]) if len(cols) > 11 else None
        # 将总投注额映射到 sales，便于兼容旧字段语义
        sales = _to_number(cols[14]) if len(cols) > 14 else None

        draws.append(
            Draw(
                issue=issue,
                draw_date=draw_date,
                reds=reds,
                blue=blue,
                sales=sales,
                pool=pool,
                first_prize_count=first_prize_count,
                first_prize_amount=first_prize_amount,
            )
        )

    return draws


def fetch_all_draws() -> List[Draw]:
    draw_map: Dict[str, Draw] = {}

    # 1) 先用福彩官网接口获取近 2000 期（约 2013 至今）
    try:
        resp = requests.get(
            CWL_API,
            params={"name": "ssq", "issueCount": 5000},
            headers=DEFAULT_HEADERS,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("state") == 0:
            for item in data.get("result", []):
                issue = str(item.get("code"))
                date_str = str(item.get("date", ""))[:10]
                try:
                    draw_date = datetime.strptime(date_str, "%Y-%m-%d").date().isoformat()
                except ValueError:
                    continue

                reds = [int(x) for x in str(item.get("red", "")).split(",") if x]
                blue = int(str(item.get("blue", "0")) or 0)

                sales = _to_number(str(item.get("sales", "")).replace(",", ""))
                pool = _to_number(str(item.get("poolmoney", "")).replace(",", ""))

                prize = item.get("prizegrades") or []
                first = prize[0] if prize else {}
                first_prize_count = _to_int(str(first.get("typenum", "")))
                first_prize_amount = _to_number(str(first.get("typemoney", "")).replace(",", ""))

                d = Draw(
                    issue=issue,
                    draw_date=draw_date,
                    reds=reds,
                    blue=blue,
                    sales=sales,
                    pool=pool,
                    first_prize_count=first_prize_count,
                    first_prize_amount=first_prize_amount,
                )
                draw_map[issue] = d
    except Exception:
        # 保底用 500.com
        pass

    # 2) 用 500.com 补齐 2010-2012（五位期号：10001-12999）
    for year in (2010, 2011, 2012):
        start = f"{year % 100:02d}001"
        end = f"{year % 100:02d}400"
        try:
            url = f"https://datachart.500.com/ssq/history/inc/history.php?start={start}&end={end}"
            html = fetch_history_html(url)
            for d in parse_history_500(html):
                if d.issue not in draw_map:
                    draw_map[d.issue] = d
        except Exception:
            continue

    # 3) 兼容更早数据（2003-2009）从 500.com 抓取
    try:
        html = fetch_history_html()
        legacy = parse_history_500(html)
        for d in legacy:
            if d.issue not in draw_map:
                draw_map[d.issue] = d
    except Exception:
        pass

    # 输出按日期排序
    return sorted(draw_map.values(), key=lambda d: d.draw_date)


def filter_new_draws(draws: Iterable[Draw], existing_issues: Set[str]) -> List[Draw]:
    today = datetime.now().date()
    fresh: List[Draw] = []
    for d in draws:
        if d.issue in existing_issues:
            continue
        try:
            draw_day = datetime.strptime(d.draw_date, "%Y-%m-%d").date()
        except ValueError:
            continue
        if draw_day <= today:
            fresh.append(d)
    return fresh

