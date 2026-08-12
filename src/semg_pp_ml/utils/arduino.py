"""
Arduino sketch generation for embedded deployment.
"""
from __future__ import annotations

from typing import Dict, Optional
from pathlib import Path


ARDUINO_SKETCH_TEMPLATE = r'''
/*
 * ============================================
 * sEMG Gesture Classifier - Seeed XIAO nRF52840
 * ============================================
 * Board: Seeed XIAO nRF52840 Sense
 * Model: TFLite INT8 quantized
 * Input: {n_channels}-channel sEMG, {window_size} timesteps
 * Output: {n_classes} classes
 */

#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "{model_name}_data.h"

constexpr int kTensorArenaSize = {tensor_arena_kb} * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

constexpr int N_CHANNELS = {n_channels};
constexpr int WINDOW_SIZE = {window_size};
constexpr int N_CLASSES = {n_classes};
const int EMG_PINS[N_CHANNELS] = {{{emg_pins}}};

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

void setup() {{
  Serial.begin(115200);
  while (!Serial);
  delay(1000);
  Serial.println("=== sEMG TinyML nRF52840 ===");

  model = tflite::GetModel({model_name}_tflite);
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {{
    Serial.println("ERROR: AllocateTensors failed!");
    return;
  }}
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  Serial.println("Model ready!");
}}

int8_t float_to_int8(float val) {{
  float scaled = val * {scale_factor};
  if (scaled > 127) return 127;
  if (scaled < -128) return -128;
  return (int8_t)scaled;
}}

void loop() {{
  int16_t raw_buffer[WINDOW_SIZE][N_CHANNELS];
  for (int t = 0; t < WINDOW_SIZE; t++) {{
    unsigned long t0 = micros();
    for (int ch = 0; ch < N_CHANNELS; ch++) {{
      raw_buffer[t][ch] = analogRead(EMG_PINS[ch]);
    }}
    while (micros() - t0 < {sample_interval_us}) {{}}
  }}

  for (int t = 0; t < WINDOW_SIZE; t++) {{
    for (int ch = 0; ch < N_CHANNELS; ch++) {{
      float volt = raw_buffer[t][ch] * (3.3 / 1023.0);
      input_tensor->data.int8[t * N_CHANNELS + ch] = float_to_int8(volt);
    }}
  }}

  unsigned long t0 = micros();
  TfLiteStatus status = interpreter->Invoke();
  unsigned long inf_us = micros() - t0;

  if (status != kTfLiteOk) {{
    Serial.println("Inference failed!");
    return;
  }}

  int8_t* out = output_tensor->data.int8;
  float scale = output_tensor->params.scale;
  int zp = output_tensor->params.zero_point;

  int best_class = 0;
  float best_prob = -999;
  Serial.print("Probs: ");
  for (int i = 0; i < N_CLASSES; i++) {{
    float prob = (out[i] - zp) * scale;
    Serial.print(prob, 3); Serial.print(" ");
    if (prob > best_prob) {{ best_prob = prob; best_class = i; }}
  }}
  Serial.println();
  Serial.print("Pred: "); Serial.print(best_class);
  Serial.print(" | "); Serial.print(inf_us); Serial.println(" us");
  Serial.println("--------------------");
  delay({loop_delay_ms});
}}
'''


def generate_arduino_sketch(
    config: Dict,
    model_name: str = "tiny_nrf52840",
    output_path: Optional[str | Path] = None,
    tensor_arena_kb: int = 60,
    scale_factor: float = 50.0,
    sample_interval_us: int = 500,
    loop_delay_ms: int = 100
) -> str:
    """
    Generate Arduino sketch for Seeed XIAO nRF52840.

    Args:
        config: Dict with n_channels, window_size, n_classes, emg_pins (optional)
        model_name: Model name for TFLite header
        output_path: Output file path
        tensor_arena_kb: Tensor arena size in KB
        scale_factor: Scaling factor for float to int8 conversion
        sample_interval_us: Sampling interval in microseconds
        loop_delay_ms: Delay between inferences in ms

    Returns:
        Sketch content as string
    """
    n_channels = config.get('n_channels', 6)
    window_size = config.get('window_size', 400)
    n_classes = config.get('n_classes', 6)
    emg_pins = config.get('emg_pins', [f"A{i}" for i in range(n_channels)])
    pins_str = ", ".join(emg_pins)

    sketch = ARDUINO_SKETCH_TEMPLATE.format(
        n_channels=n_channels,
        window_size=window_size,
        n_classes=n_classes,
        model_name=model_name,
        emg_pins=pins_str,
        tensor_arena_kb=tensor_arena_kb,
        scale_factor=scale_factor,
        sample_interval_us=sample_interval_us,
        loop_delay_ms=loop_delay_ms
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(sketch)
        print(f"[OK] Arduino sketch: {output_path}")

    return sketch


def save_arduino_sketch(config: Dict, model_name: str = "tiny_nrf52840", output_dir: str | Path = "outputs") -> str:
    """Save Arduino sketch to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sketch_path = output_dir / f"{model_name}_arduino.ino"
    return generate_arduino_sketch(config, model_name, sketch_path)