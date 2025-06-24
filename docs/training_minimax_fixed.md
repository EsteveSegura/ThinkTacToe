# Entrenamiento SFT Corregido para Formato Minimax

## 🎯 Problema Resuelto

El script original tenía varios problemas que causaban que el modelo no aprendiera el formato correcto:

1. **Token EOS incorrecto**: `<|im_end|>` en lugar de `<|end|>`
2. **Dataset incorrecto**: Usaba un dataset diferente al minimax
3. **Configuración subóptima**: Batch size y learning rate no optimizados

## 🚀 Comando de Entrenamiento

```bash
accelerate launch src/train/train_sft_minimax_fixed.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --data_files "./datasets/tictactoe_minimax_20250624_205741.jsonl" \
    --output_dir "qwen2.5-0.5b-tictactoe-sft-minimax-fixed" \
    --logs_dir "./logs" \
    --num_epochs 2 \
    --batch_size 8 \
    --learning_rate 5e-5
```

## ⚙️ Configuración Optimizada

- **EOS Token**: `<|end|>` (corregido)
- **Max Sequence Length**: 512 (aumentado para formato estructurado)
- **Batch Size**: 8 (reducido para estabilidad)
- **Learning Rate**: 5e-5 (optimizado)
- **Épocas**: 2 (suficiente para aprender el formato)

## 🔍 Validación del Dataset

El script incluye validación automática que verifica:
- ✅ Presencia de todos los tokens especiales
- ✅ Formato correcto de movimientos: `<|move|><|0-0|><|end|>`
- ✅ Estructura completa del dataset

## 📊 Resultados Esperados

Después del entrenamiento, el modelo debería generar:
```
<|move|><|1-0|><|end|>
```

En lugar de:
```
0-0
```

## 🧪 Testing

Después del entrenamiento, usar:
```bash
python src/inference/inference_sft_minimax_fixed.py
```

Para verificar que el modelo genera el formato correcto. 