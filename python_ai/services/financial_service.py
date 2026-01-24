"""
Financial/Fundamental Analysis Service
Calculates fundamental score and metrics from real financial ratios using Yahoo Finance
"""

import yfinance as yf
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAnalyzer:
    """
    Financial/Fundamental Analysis Service
    Provides fundamental score and metrics for a given ticker
    """

    def fetch_score(self, ticker: str) -> float:
        """
        Calculate comprehensive fundamental score (0-1)
        Based on multiple financial ratios and metrics
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            scores = []
            weights = []

            # ==================== VALUATION METRICS ====================
            pe_ratio = info.get('trailingPE', info.get('forwardPE'))
            if pe_ratio and pe_ratio > 0:
                if pe_ratio < 10:
                    pe_score = 0.5
                elif pe_ratio <= 20:
                    pe_score = 1.0
                elif pe_ratio <= 30:
                    pe_score = 0.7
                else:
                    pe_score = 0.4
                scores.append(pe_score)
                weights.append(1.5)

            pb_ratio = info.get('priceToBook')
            if pb_ratio and pb_ratio > 0:
                if pb_ratio < 1:
                    pb_score = 1.0
                elif pb_ratio <= 2:
                    pb_score = 0.9
                elif pb_ratio <= 3:
                    pb_score = 0.7
                else:
                    pb_score = 0.5
                scores.append(pb_score)
                weights.append(1.0)

            # ==================== PROFITABILITY METRICS ====================
            roe = info.get('returnOnEquity')
            if roe is not None:
                roe_pct = roe * 100
                if roe_pct > 20:
                    roe_score = 1.0
                elif roe_pct > 15:
                    roe_score = 0.9
                elif roe_pct > 10:
                    roe_score = 0.7
                elif roe_pct > 5:
                    roe_score = 0.5
                else:
                    roe_score = 0.3
                scores.append(roe_score)
                weights.append(1.5)

            profit_margin = info.get('profitMargins')
            if profit_margin is not None:
                pm_pct = profit_margin * 100
                if pm_pct > 20:
                    pm_score = 1.0
                elif pm_pct > 15:
                    pm_score = 0.9
                elif pm_pct > 10:
                    pm_score = 0.7
                elif pm_pct > 5:
                    pm_score = 0.5
                else:
                    pm_score = 0.3
                scores.append(pm_score)
                weights.append(1.0)

            operating_margin = info.get('operatingMargins')
            if operating_margin is not None:
                om_pct = operating_margin * 100
                if om_pct > 20:
                    om_score = 1.0
                elif om_pct > 15:
                    om_score = 0.8
                elif om_pct > 10:
                    om_score = 0.6
                else:
                    om_score = 0.4
                scores.append(om_score)
                weights.append(0.8)

            # ==================== LEVERAGE METRICS ====================
            debt_to_equity = info.get('debtToEquity')
            if debt_to_equity is not None:
                if debt_to_equity < 30:
                    de_score = 1.0
                elif debt_to_equity < 50:
                    de_score = 0.9
                elif debt_to_equity < 100:
                    de_score = 0.7
                elif debt_to_equity < 150:
                    de_score = 0.5
                else:
                    de_score = 0.3
                scores.append(de_score)
                weights.append(1.5)

            # ==================== GROWTH METRICS ====================
            earnings_growth = info.get('earningsQuarterlyGrowth')
            if earnings_growth is not None:
                if earnings_growth > 0.3:
                    eg_score = 1.0
                elif earnings_growth > 0.2:
                    eg_score = 0.9
                elif earnings_growth > 0.1:
                    eg_score = 0.7
                elif earnings_growth > 0:
                    eg_score = 0.5
                else:
                    eg_score = 0.2
                scores.append(eg_score)
                weights.append(1.5)

            revenue_growth = info.get('revenueGrowth')
            if revenue_growth is not None:
                if revenue_growth > 0.3:
                    rg_score = 1.0
                elif revenue_growth > 0.2:
                    rg_score = 0.9
                elif revenue_growth > 0.1:
                    rg_score = 0.7
                elif revenue_growth > 0:
                    rg_score = 0.5
                else:
                    rg_score = 0.2
                scores.append(rg_score)
                weights.append(1.0)

            # ==================== LIQUIDITY METRICS ====================
            current_ratio = info.get('currentRatio')
            if current_ratio:
                if 1.5 <= current_ratio <= 3:
                    cr_score = 1.0
                elif 1 <= current_ratio < 1.5:
                    cr_score = 0.7
                elif 3 < current_ratio <= 4:
                    cr_score = 0.7
                else:
                    cr_score = 0.4
                scores.append(cr_score)
                weights.append(0.8)

            quick_ratio = info.get('quickRatio')
            if quick_ratio:
                if quick_ratio >= 1:
                    qr_score = 1.0
                elif quick_ratio >= 0.5:
                    qr_score = 0.6
                else:
                    qr_score = 0.3
                scores.append(qr_score)
                weights.append(0.6)

            # ==================== DIVIDEND METRICS ====================
            dividend_yield = info.get('dividendYield')
            if dividend_yield:
                dy_pct = dividend_yield * 100
                if dy_pct > 5:
                    dy_score = 1.0
                elif dy_pct > 3:
                    dy_score = 0.8
                elif dy_pct > 1:
                    dy_score = 0.6
                else:
                    dy_score = 0.4
                scores.append(dy_score)
                weights.append(0.5)

            # ==================== FINAL SCORE CALCULATION ====================
            if scores and weights:
                weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
                final_score = float(max(0.0, min(1.0, weighted_score)))
                logger.info(f"  📊 {ticker} Fundamental Score: {final_score:.2f}")
                return final_score

            logger.warning(f"  ⚠️  {ticker} - No financial data available")
            return 0.5

        except Exception as e:
            logger.error(f"  ✗ {ticker} - Financial score error: {str(e)}")
            return 0.5

    def get_metrics(self, ticker: str) -> Dict:
        """
        Return all available financial metrics as dictionary
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                'pe_ratio': info.get('trailingPE', info.get('forwardPE')),
                'pb_ratio': info.get('priceToBook'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'peg_ratio': info.get('pegRatio'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'gross_margin': info.get('grossMargins'),
                'debt_to_equity': info.get('debtToEquity'),
                'debt_to_assets': info.get('totalDebt') / info.get('totalAssets') if info.get('totalDebt') and info.get('totalAssets') else None,
                'earnings_growth': info.get('earningsQuarterlyGrowth'),
                'revenue_growth': info.get('revenueGrowth'),
                'current_ratio': info.get('currentRatio'),
                'quick_ratio': info.get('quickRatio'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'market_cap': info.get('marketCap'),
                'beta': info.get('beta'),
                'eps': info.get('trailingEps'),
                'book_value': info.get('bookValue'),
            }

        except Exception as e:
            logger.error(f"Error getting financial metrics for {ticker}: {str(e)}")
            return {}
