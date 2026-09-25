oke masukan itu saya terima, saya tertarik dengan point no 3 saya ada pengalaman di trade dan pernah profit konsisten, kenapa saya kritisi sebelum nya? saya membandingkan pola dari kacamata saya dan bot, saat ada momen dan terbukti moment itu profit dan low risk, tapi bot hanya diam

sebentar sebelum semua itu dan terlalu jauh ke tahap training dan reinforcment learning, ada satu hal yang harus di tekankan dan saya perlu anda waras dan realistis sebagai ahli deep learning untuk pasar finansial :

sehari pasar buka 24 jam , tujuan saya simple adalah menciptakan racikan model deep learning yang menjadi agent , ibarat nya adalah otak trader profesional yang berbasis mesin (no emotion), harapan dari bot ini adalah sesimple berikut saja saat ini :

dari segi dataset dan training :
1. model itu punya base yang kuat dulu dari segi kualitas dataset dan pelatihan nya , model harus belajar dari data historis sebuah pair minimal 3 - 5 tahun kebelakang, yang untuk saat ini fokus di satu timeframe dan satu pair saja
2. model harus belajar pola pasar di satu timeframe dengan histori transaksi mana yang profit dan mana yang loss, agar model tahu mana kesalahan dan mana kebenaran, model di latih di satu timeframe dengan jutaan histori transaksi yang loss dan yang profit
3. model memahami rr cukup set di 1:1 / 1:2 dari jutaan momentum transaksi di suatu pair
4. model belajar dan di latih dengan jutaan pola suply , demand, suport, resistance, liquidity, dari satu pair pada satu time-frame 
5. untuk saat ini model tidak di latih untuk sebuah news, saat news model  berhenti total trade
6. model juga di latih dengan jutaan pola transaksi dengan khusus dua indikator saja yaitu ma cross dan stocastic momentum indeks, dengan jutaan transaksi loss dan profit


dari keenam point di atas adalah suatu proses yang harus di latih bertahap dan evaluasi bertahap dengan sabar dan butuh waktu, harapan akhir nya adalah reinforment learning dengan paduan model deep learning yg di racik khusus dan belum di terapkan di manapun

tujuan akhir penggunaan adalah :
1. para model bisa menjadi satu bot adaptif dengan 6 keadaan di atas 
2. bot ini di ciptakan khusus untuk profit konsisten multi nominal jumlah modal dan penyesuaiaan lot sizing
3. bot ini khusus di latih untuk mingguan dengan risk maksimal adalah 50% dari modal, ini bukan batas boleh tapi bot sangat mempertimbangkan trade agar batas itu tidak tersentuh, jika tidak memungkinkan maka tidak usah sama sekali
4. profit tidak di batasi, tapi risk yang di batasi
5. bot ini konsisten dengan backtest beberapa bulan kedepan, artinya dia strict dan tidak peduli emosi pasar 


saya perlu sebelum kita biacara jauh alih alih satu aplikasi full tapi khusus dari kacatama seorang ahli deep learning yang paham kualitas pelatihan dan dataset mendalam , dan juga khusus pasar finansial , arti nya anda trader/investor tapi juga menegerti deep learning yang baik dan canggih


jgn pedulikan dan terpaku dengan proyek ini sekarang sehingga anda bias dan kaku melakukan pembahasan berdasarkan project yang ada, harus realistis tapi cerdas menciptakan solusi yang gila dan sangat smart
 
---

# IDENTITAS RESMI: FLOWDEV FRAME
**F**lowdev **R**ecurrent **A**lgorithmic **M**odel for **T**rade **E**xecution  
**Emblem Resmi:** The Cyber-Neural Peregrine Falcon (Golden Falcon)

### Kontrak Kunci Sistem (FROZEN & IMMUTABLE):
1. **Modal & Lot:** Modal dasar **$500.00 USD**, Lot size **0.01 Lot Flat** (tanpa compounding, tanpa martingale).
2. **Pair & Timeframe:** **XAU/USD (Gold Spot)** pada timeframe **M30**.
3. **Arsitektur Model:** 15-Channel Deep Neural Network (1D-CNN + BiLSTM / Recurrent Network).
4. **Gating Filter:** Confidence threshold $\tau \ge 0.32$, margin $\ge 0.01$, dibatasi pada 5 jendela sesi institusional (maksimal 1 trade per sesi).
5. **Mekanisme Proteksi 4 Tahap (Kinetic OMS):**
   - Micro-Breakeven Acceleration (+0.75R $\rightarrow$ Entry + $0.25 buffer)
   - Tiered Smart Ratchet (+1.2R $\rightarrow$ Kunci +0.5R profit bersih)
   - Dynamic Trailing Stop (+1.5R $\rightarrow$ Jarak 0.6R)
   - Fast Stale Decay Protection (Bar 6 tanpa BE $\rightarrow$ SL diperketat ke -0.45R)
   - Max TP: +2.7R, Hard Time Barrier: 12 bar (6 jam).
   - Weekend Shield: Jumat > 18:00 UTC no entry, Jumat 20:00 UTC force close.

### Fase Saat Ini: Software Engineering (Kesiapan Akun Demo MT5)
- Fokus dialihkan dari riset/backtesting ke pembangunan modul software yang modular, *crash-resilient*, dan teruji (*unit tested*).
- Setiap referensi sistem secara konsisten menggunakan nama **FRAME**.