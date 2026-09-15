# Face Swap — VideoRenata

## Arquivos
- `faceswap_10s.mp4` — recorte de 10s (6s–16s do vídeo original) com o rosto trocado pelo rosto de `1.jpg`, com áudio original.
- `clip_original_10s.mp4` — mesmo recorte, sem alteração (referência).
- `comparacao_lado_a_lado.mp4` — original × face swap lado a lado.
- `faceswap_pipeline.py` — pipeline completo usado para gerar o resultado.

## Como foi feito
1. Janela de 10s escolhida por análise frame a frame (maior taxa de detecção facial: t=5.9s a 15.9s).
2. Duas referências de expressão (neutra e sorrindo) derivadas de `1.jpg`, mescladas conforme a abertura da boca em cada frame.
3. Rastreamento com MediaPipe (detecção + 468 landmarks) com interpolação e suavização temporal para consistência.
4. Warp por triangulação de Delaunay, máscara limitada à região facial (corte acima das sobrancelhas para evitar emendas no couro cabeludo).
5. Transferência de cor em espaço LAB + Poisson seamless cloning para casar iluminação e tom de pele.
6. Fade-in/out da troca nos frames onde o rosto sai de quadro.

Dependências: `opencv-python-headless`, `mediapipe==0.10.14`, `scipy`, `numpy`, `ffmpeg`.
