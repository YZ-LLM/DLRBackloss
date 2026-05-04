# DLR-BackLoss: Dynamic Linear Recurrent BackLoss

Bu proje, Transformer mimarisinin $O(n^2)$ karmaşıklığına alternatif olarak geliştirilen, lineer zaman karmaşıklığına ($O(n)$) sahip hibrit bir derin öğrenme katmanıdır.

### 1. BackLoss Formülasyonu
DLR-BackLoss, her zaman adımında geçmiş bilgilerin kümülatif istatistiklerini hesaplayarak 'yenilik' ve 'ilgi' sinyallerini belirler. Modelin seçicilik katsayısı (gating) şu formüle dayanır:

$$\alpha_t = \sigma(Q_t - \mu_{k, 1:t})$$

Burada:
*   **$\mu_{k, 1:t}$**: $t$ zamanına kadar olan tüm anahtar ($K$) değerlerinin kümülatif ortalamasıdır.
*   **$Q_t - \mu$**: Sorgunun, geçmişin genel karakterinden ne kadar saptığını ölçer (Surprise Signal).
*   **$\\sigma$**: Sigmoid aktivasyonu ile 0-1 arasına sıkıştırılan seçicilik katsayısıdır.

### 2. Performans ve Benchmarks (Zeka Verimliliği)
Yaptığımız testlerde, modelin donanım bağımsız başarısı "Intelligence per Second" (Birim Saniyede Kazanılan Doğruluk) metriği ile ölçülmüştür.

| Cihaz | Model | İterasyon (60s) | Doğruluk (Acc) | Verimlilik Katsayısı |
| :--- | :--- | :--- | :--- | :--- |
| **CPU** | Standard Transformer | 138 | %10.89 | 0.181 |
| **CPU** | **DLR-BackLoss** | **2200** | **%13.58** | **0.226** |
| **GPU (T4)** | Standard Transformer | 1851 | %7.33 | 0.122 |
| **GPU (T4)** | **DLR-BackLoss** | **9703** | **%12.10** | **0.201** |
| **GPU (SOTA)** | Mamba (Parallel) | 2092 | %7.93 | 0.132 |
| **GPU (SOTA)** | **DLR-BackLoss** | **2096** | **%10.34** | **0.172** |

### 3. Temel Avantajlar
*   **$O(n)$ Ölçeklenebilirlik:** Dizi uzunluğu karesel değil, doğrusal artar. Bu sayede çok uzun metinlerde (Context Window) Transformer'dan kat kat hızlıdır.
*   **Donanım Verimliliği:** Vektörize edilmiş kümülatif toplamlar (`torch.cumsum`) sayesinde GPU çekirdeklerini en az matris çarpımı kadar verimli kullanır.
*   **Dinamik Hafıza:** BackLoss sinyali, modelin neyi unutup neyi hatırlayacağını sabit ağırlıklarla değil, verinin o anki istatistiksel sapmalarıyla belirler.

### 4. Kullanım Alanları
*   Düşük donanımlı (Edge) cihazlarda yüksek performanslı dil işleme.
*   Çok uzun dizi (Time-series) verilerinde örüntü yakalama.
*   Transformer tabanlı LLM'lerde verimlilik artırıcı hibrit katmanlar.
