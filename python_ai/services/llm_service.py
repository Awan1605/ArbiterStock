# File: python_ai/services/llm_service.py

"""
LLM Service - Template-based Analysis (No External API Required)
Analisis saham profesional dalam Bahasa Indonesia untuk pemula
UPDATED: Support untuk prediksi harian 30 hari
"""

from typing import Dict, Any, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class LLMService:
    """
    Local LLM Service menggunakan rule-based template
    Tidak memerlukan API key atau token eksternal
    """
    
    def __init__(self):
        logger.info("🤖 LLM Service initialized (Template-based)")
    
    def _get_recommendation(self, combined_score: float) -> tuple[str, str, str]:
        """Generate rekomendasi berdasarkan combined score"""
        if combined_score >= 80:
            return "BELI KUAT", "Sangat Positif", "Tinggi"
        elif combined_score >= 70:
            return "BELI", "Positif", "Tinggi"
        elif combined_score >= 60:
            return "BELI (Hati-hati)", "Cukup Positif", "Sedang"
        elif combined_score >= 50:
            return "TAHAN", "Netral", "Sedang"
        elif combined_score >= 40:
            return "JUAL (Hati-hati)", "Cukup Negatif", "Sedang"
        elif combined_score >= 30:
            return "JUAL", "Negatif", "Tinggi"
        else:
            return "JUAL KUAT", "Sangat Negatif", "Tinggi"
    
    def _analyze_rsi(self, rsi: float) -> str:
        """Analisis indikator RSI"""
        if rsi > 70:
            return f"RSI {rsi:.0f} - Kondisi jenuh beli (overbought). Harga kemungkinan akan turun."
        elif rsi < 30:
            return f"RSI {rsi:.0f} - Kondisi jenuh jual (oversold). Harga kemungkinan akan naik."
        elif 45 <= rsi <= 55:
            return f"RSI {rsi:.0f} - Kondisi seimbang. Tidak ada sinyal kuat."
        else:
            return f"RSI {rsi:.0f} - Kondisi normal."
    
    def _analyze_macd(self, macd: float, macd_signal: float) -> str:
        """Analisis indikator MACD"""
        if macd > macd_signal:
            diff = abs(macd - macd_signal)
            if diff > 0.5:
                return "MACD menunjukkan momentum naik kuat (bullish)."
            else:
                return "MACD menunjukkan tren naik sedang."
        elif macd < macd_signal:
            diff = abs(macd - macd_signal)
            if diff > 0.5:
                return "MACD menunjukkan momentum turun kuat (bearish)."
            else:
                return "MACD menunjukkan tren turun sedang."
        else:
            return "MACD netral, belum ada sinyal jelas."
    
    def _analyze_moving_averages(self, current_price: float, sma_20: float, 
                                  sma_50: float, sma_200: float) -> str:
        """Analisis moving average"""
        signals = []
        
        if current_price > sma_20:
            signals.append("di atas MA 20 hari (tren naik jangka pendek)")
        else:
            signals.append("di bawah MA 20 hari (tren turun jangka pendek)")
            
        if current_price > sma_50:
            signals.append("di atas MA 50 hari (tren naik jangka menengah)")
        else:
            signals.append("di bawah MA 50 hari (tren turun jangka menengah)")
            
        if current_price > sma_200:
            signals.append("di atas MA 200 hari (tren naik jangka panjang)")
        else:
            signals.append("di bawah MA 200 hari (tren turun jangka panjang)")
        
        # Golden Cross / Death Cross
        if sma_50 > sma_200:
            signals.append("Golden Cross terdeteksi (sinyal beli kuat)")
        elif sma_50 < sma_200:
            signals.append("Death Cross terdeteksi (sinyal jual kuat)")
        
        return "Harga " + ", ".join(signals) + "."
    
    def _get_trend_label(self, score: float) -> str:
        """Label tren berdasarkan score"""
        if score >= 70:
            return "Kuat"
        elif score >= 60:
            return "Cukup Baik"
        elif score >= 50:
            return "Netral"
        elif score >= 40:
            return "Lemah"
        else:
            return "Sangat Lemah"
    
    def _format_change(self, change_pct: float) -> tuple[str, str]:
        """Format perubahan harga dengan warna dan deskripsi"""
        if abs(change_pct) < 2:
            return "stabil", "📊"
        elif change_pct > 10:
            return "naik signifikan", "📈"
        elif change_pct > 5:
            return "naik kuat", "📈"
        elif change_pct > 0:
            return "naik", "📈"
        elif change_pct < -10:
            return "turun signifikan", "📉"
        elif change_pct < -5:
            return "turun kuat", "📉"
        else:
            return "turun", "📉"
    
    def _get_risk_level(self, confidence: float) -> tuple[str, str]:
        """Tentukan tingkat risiko berdasarkan confidence"""
        if confidence >= 85:
            return "RENDAH", "Model sangat yakin dengan prediksi ini."
        elif confidence >= 70:
            return "SEDANG", "Model cukup yakin dengan prediksi ini."
        elif confidence >= 60:
            return "TINGGI", "Model kurang yakin, prediksi bisa meleset."
        else:
            return "SANGAT TINGGI", "Model tidak yakin, hindari keputusan berdasarkan prediksi ini."
    
    def _analyze_prediction_trend(self, daily_predictions: List[Dict]) -> tuple[str, float, str]:
        """Analyze the trend from daily predictions"""
        if not daily_predictions or len(daily_predictions) < 2:
            return "Tidak tersedia", 0.0, "📊"
        
        first_price = daily_predictions[0]['price']
        last_price = daily_predictions[-1]['price']
        
        change_pct = ((last_price - first_price) / first_price) * 100
        
        # Count upward and downward days
        upward_days = 0
        downward_days = 0
        
        for i in range(1, len(daily_predictions)):
            if daily_predictions[i]['price'] > daily_predictions[i-1]['price']:
                upward_days += 1
            elif daily_predictions[i]['price'] < daily_predictions[i-1]['price']:
                downward_days += 1
        
        # Determine trend
        if upward_days > downward_days * 1.5:
            trend = "Bullish (Naik Konsisten)"
            icon = "📈"
        elif downward_days > upward_days * 1.5:
            trend = "Bearish (Turun Konsisten)"
            icon = "📉"
        elif abs(change_pct) < 3:
            trend = "Sideways (Bergerak Mendatar)"
            icon = "↔️"
        else:
            trend = "Volatil (Naik-Turun)"
            icon = "📊"
        
        return trend, change_pct, icon
    
    def _get_action_plan(self, recommendation: str, technical_score: float, 
                         fundamental_score: float, trend: str, change_pct: float) -> str:
        """Generate action plan yang actionable"""
        tech_label = self._get_trend_label(technical_score)
        fund_label = self._get_trend_label(fundamental_score)
        
        if recommendation in ["BELI KUAT", "BELI"]:
            if technical_score > 70 and fundamental_score > 70:
                return f"Rekomendasi: Beli bertahap saat harga turun. Teknikal {tech_label} + Fundamental {fund_label} = Peluang bagus untuk investasi jangka menengah-panjang. Tren prediksi {trend} mendukung keputusan beli."
            elif technical_score > 70:
                return f"Rekomendasi: Beli untuk trading jangka pendek. Teknikal {tech_label} tapi fundamental {fund_label}. Pasang target profit dan stop loss. Monitor tren {trend} dalam 30 hari ke depan."
            else:
                return f"Rekomendasi: Beli untuk investasi jangka panjang. Fundamental {fund_label} tapi teknikal {tech_label}. Tunggu koreksi untuk entry yang lebih baik."
                
        elif recommendation in ["JUAL KUAT", "JUAL"]:
            if technical_score < 40 and fundamental_score < 40:
                return f"Rekomendasi: Jual atau kurangi posisi. Teknikal {tech_label} + Fundamental {fund_label} = Risiko tinggi untuk kerugian lebih lanjut. Tren {trend} menunjukkan potensi penurunan."
            elif technical_score < 40:
                return f"Rekomendasi: Pasang stop loss ketat. Teknikal {tech_label} menunjukkan risiko penurunan. Fundamental {fund_label} bisa recovery tapi butuh waktu."
            else:
                return f"Rekomendasi: Jual sebagian untuk ambil profit. Fundamental {fund_label} tapi teknikal masih support. Amankan keuntungan yang sudah ada."
                
        else:  # TAHAN
            if "Sideways" in trend:
                return f"Rekomendasi: Tahan posisi dan monitor. Market sedang sideways (bergerak mendatar). Teknikal {tech_label}, Fundamental {fund_label}. Tunggu sinyal jelas sebelum action."
            else:
                return f"Rekomendasi: Tahan posisi saat ini. Belum ada sinyal kuat untuk beli atau jual. Teknikal {tech_label}, Fundamental {fund_label}. Pantau perkembangan market."
    
    def _format_prediction_summary(self, daily_predictions: List[Dict], current_price: float) -> str:
        """Format summary prediksi mingguan dari data harian"""
        if not daily_predictions:
            return "Data prediksi tidak tersedia"
        
        # Week 1 (Day 7)
        week1 = daily_predictions[6] if len(daily_predictions) > 6 else daily_predictions[-1]
        week1_change = ((week1['price'] - current_price) / current_price) * 100
        week1_trend, week1_icon = self._format_change(week1_change)
        
        # Week 2 (Day 14)
        week2 = daily_predictions[13] if len(daily_predictions) > 13 else daily_predictions[-1]
        week2_change = ((week2['price'] - current_price) / current_price) * 100
        week2_trend, week2_icon = self._format_change(week2_change)
        
        # Week 3 (Day 21)
        week3 = daily_predictions[20] if len(daily_predictions) > 20 else daily_predictions[-1]
        week3_change = ((week3['price'] - current_price) / current_price) * 100
        week3_trend, week3_icon = self._format_change(week3_change)
        
        # Week 4 (Day 30)
        week4 = daily_predictions[29] if len(daily_predictions) > 29 else daily_predictions[-1]
        week4_change = ((week4['price'] - current_price) / current_price) * 100
        week4_trend, week4_icon = self._format_change(week4_change)
        
        return f"""{week1_icon} Minggu 1 (Hari ke-7)  : ${week1['price']:,.2f} ({week1_change:+.1f}%) - {week1_trend}
{week2_icon} Minggu 2 (Hari ke-14) : ${week2['price']:,.2f} ({week2_change:+.1f}%) - {week2_trend}
{week3_icon} Minggu 3 (Hari ke-21) : ${week3['price']:,.2f} ({week3_change:+.1f}%) - {week3_trend}
{week4_icon} Minggu 4 (Hari ke-30) : ${week4['price']:,.2f} ({week4_change:+.1f}%) - {week4_trend}"""
    
    def generate_analysis(self, ticker: str, data: Dict[str, Any]) -> str:
        """
        Generate analisis saham komprehensif dalam Bahasa Indonesia
        
        Args:
            ticker: Simbol ticker saham
            data: Dictionary berisi semua metrik saham
            
        Returns:
            str: Teks analisis terformat
        """
        try:
            # Extract data
            current_price = data.get('current_price', 0)
            daily_predictions = data.get('daily_predictions', [])
            
            rsi = data.get('rsi', 50)
            macd = data.get('macd', 0)
            macd_signal = data.get('macd_signal', 0)
            sma_20 = data.get('sma_20', 0)
            sma_50 = data.get('sma_50', 0)
            sma_200 = data.get('sma_200', 0)
            
            technical_score = data.get('technical_score', 50)
            fundamental_score = data.get('fundamental_score', 50)
            news_score = data.get('news_score', 0)
            combined_score = data.get('combined_score', 50)
            confidence = data.get('confidence', 50)
            
            # Analyze prediction trend
            trend, overall_change, trend_icon = self._analyze_prediction_trend(daily_predictions)
            
            # Generate components
            recommendation, sentiment, conf_level = self._get_recommendation(combined_score)
            rsi_analysis = self._analyze_rsi(rsi)
            macd_analysis = self._analyze_macd(macd, macd_signal)
            ma_analysis = self._analyze_moving_averages(current_price, sma_20, sma_50, sma_200)
            risk_level, risk_desc = self._get_risk_level(confidence)
            action_plan = self._get_action_plan(recommendation, technical_score, fundamental_score, trend, overall_change)
            
            # Format weekly summary from daily predictions
            prediction_summary = self._format_prediction_summary(daily_predictions, current_price)
            
            # News sentiment
            if news_score > 60:
                news_sentiment = "sangat positif"
                news_impact = "Berita baik bisa mendorong harga naik."
            elif news_score > 45:
                news_sentiment = "positif"
                news_impact = "Berita cukup baik untuk support harga."
            elif news_score >= 35:
                news_sentiment = "netral"
                news_impact = "Berita tidak berpengaruh signifikan."
            elif news_score >= 20:
                news_sentiment = "negatif"
                news_impact = "Berita buruk bisa menekan harga."
            else:
                news_sentiment = "sangat negatif"
                news_impact = "Berita sangat buruk, hindari dulu."
            
            # Labels
            tech_label = self._get_trend_label(technical_score)
            fund_label = self._get_trend_label(fundamental_score)
            
            # Build analysis
            analysis = f"""
╔═══════════════════════════════════════════════════════════════╗
║           ANALISIS SAHAM {ticker}                                     
║           {datetime.now().strftime('%d %B %Y, %H:%M WIB')}                            
╚═══════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────┐
│ 💰 RINGKASAN INVESTASI                                      │
└─────────────────────────────────────────────────────────────┘

Harga Saat Ini    : ${current_price:,.2f}
Rekomendasi       : {recommendation}
Sentimen Market   : {sentiment}
Tingkat Keyakinan : {conf_level} ({confidence:.1f}%)
Skor Gabungan     : {combined_score:.0f}/100

┌─────────────────────────────────────────────────────────────┐
│ 🎯 PREDIKSI HARGA 30 HARI (Per Minggu)                     │
└─────────────────────────────────────────────────────────────┘

{prediction_summary}

{trend_icon} Tren Keseluruhan: {trend} ({overall_change:+.1f}% dalam 30 hari)

┌─────────────────────────────────────────────────────────────┐
│ 📊 ANALISIS TEKNIKAL (Skor: {technical_score:.0f}/100 - {tech_label})        │
└─────────────────────────────────────────────────────────────┘

• {rsi_analysis}
• {macd_analysis}
• {ma_analysis}

┌─────────────────────────────────────────────────────────────┐
│ 🏢 ANALISIS FUNDAMENTAL (Skor: {fundamental_score:.0f}/100 - {fund_label})   │
└─────────────────────────────────────────────────────────────┘

Kesehatan finansial perusahaan dinilai {fund_label.lower()}. 
{'Perusahaan memiliki fundamental kuat dengan prospek pertumbuhan bagus.' if fundamental_score > 70 else 'Perusahaan memiliki fundamental cukup stabil.' if fundamental_score > 50 else 'Perusahaan memiliki fundamental lemah, perlu perhatian khusus.'}

┌─────────────────────────────────────────────────────────────┐
│ 📰 SENTIMEN BERITA                                          │
└─────────────────────────────────────────────────────────────┘

Sentimen berita terkini: {news_sentiment.upper()} ({news_score:.0f}/100)
{news_impact}

┌─────────────────────────────────────────────────────────────┐
│ ⚠️  TINGKAT RISIKO: {risk_level}                                   │
└─────────────────────────────────────────────────────────────┘

{risk_desc}

┌─────────────────────────────────────────────────────────────┐
│ 💡 RENCANA AKSI                                             │
└─────────────────────────────────────────────────────────────┘

{action_plan}

┌─────────────────────────────────────────────────────────────┐
│ 📅 AKSES PREDIKSI HARIAN LENGKAP                            │
└─────────────────────────────────────────────────────────────┘

Prediksi harian detail untuk 30 hari tersimpan di database.
Gunakan endpoint API untuk melihat prediksi per hari secara lengkap.

╔═══════════════════════════════════════════════════════════════╗
║ ⚠️  DISCLAIMER                                                ║
║ Analisis ini hanya untuk informasi, bukan saran investasi.   ║
║ Lakukan riset sendiri dan konsultasi dengan ahli keuangan.   ║
║ Investasi mengandung risiko. Keputusan ada di tangan Anda.   ║
╚═══════════════════════════════════════════════════════════════╝
            """.strip()
            
            logger.info(f"✅ Generated analysis for {ticker}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error generating analysis for {ticker}: {e}", exc_info=True)
            return self._generate_fallback_analysis(ticker, data)
    
    def _generate_fallback_analysis(self, ticker: str, data: Dict[str, Any]) -> str:
        """Generate analisis sederhana jika terjadi error"""
        current_price = data.get('current_price', 0)
        daily_predictions = data.get('daily_predictions', [])
        combined_score = data.get('combined_score', 50)
        confidence = data.get('confidence', 50)
        
        recommendation, sentiment, _ = self._get_recommendation(combined_score)
        
        pred_30d = daily_predictions[29] if len(daily_predictions) > 29 else None
        pred_text = ""
        if pred_30d:
            change = ((pred_30d['price'] - current_price) / current_price) * 100
            pred_text = f"Prediksi 30 Hari : ${pred_30d['price']:,.2f} ({change:+.1f}%)"
        
        return f"""
╔═══════════════════════════════════════════════════════════════╗
║           ANALISIS SAHAM {ticker}                                     
╚═══════════════════════════════════════════════════════════════╝

Harga Saat Ini    : ${current_price:,.2f}
{pred_text}
Skor Gabungan     : {combined_score:.0f}/100
Tingkat Keyakinan : {confidence:.0f}%

Rekomendasi       : {recommendation}
Sentimen          : {sentiment}

⚠️  Analisis terbatas karena data tidak lengkap.
    Silakan cek detail teknikal dan fundamental secara manual.
        """.strip()
    
    async def agenerate_analysis(self, ticker: str, data: Dict[str, Any]) -> str:
        """Async version of generate_analysis"""
        return self.generate_analysis(ticker, data)


# Create singleton instance
llm_service = LLMService()


# Helper function
def generate_stock_analysis(ticker: str, stock_data: Dict[str, Any]) -> str:
    """Helper function untuk generate analysis"""
    return llm_service.generate_analysis(ticker, stock_data)