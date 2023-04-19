/**
 * Network Model Test Run
 * 
 * Author: Pete Warden
 * Modified by: 
 * Date: March 11, 2020
 * 
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Import TensorFlow stuff
#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"


/* Define the model */
#include "homeros_c_model.h"

// Figure out what's going on in our model
#define DEBUG 0

// Some settings
constexpr int led_pin = 24;
constexpr float pi = 3.14159265;                  // Some pi
constexpr float freq = 0.5;                       // Frequency (Hz) of sinewave
constexpr float period = (1 / freq) * (1000000);  // Period (microseconds)

/* TFLite globals, used for compatibility with Arduino-style sketches */
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  /* Create an area of memory to use for input, output, and other TensorFlow
  arrays. You'll need to adjust this by combiling, running, and looking
  for errors. */
  constexpr int kTensorArenaSize = 5 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
} 

void setup() {

  // Wait for Serial to connect
#if DEBUG
  while(!Serial);
#endif


  /* Map the model into a usable data structure */
  model = tflite::GetModel(homeros_c_model);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }


  /* Pull in only needed operations (should match NN layers)
  Available ops: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h */
  static tflite::AllOpsResolver resolver;

  /* Build an interpreter to run the model */
  static tflite::MicroInterpreter static_interpreter(
  model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  /* Allocate memory from the tensor_arena for the model's tensors */
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }

  /* Assign model input and output buffers (tensors) to pointers */
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

#if DEBUG
  Serial.print("Number of dimensions: ");
  Serial.println(model_input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(model_input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(model_input->dims->data[1]);
  Serial.print("Input type: ");
  Serial.println(model_input->type);
#endif
}

void loop() {

#if DEBUG
  unsigned long start_timestamp = micros();
#endif

  /* Get current timestamp and modulo with period */
  unsigned long timestamp = micros();
  timestamp = timestamp % (unsigned long)period;

  // Calculate x value to feed to the model
  float x_val = ((float)timestamp * 2 * pi) / period;

  // Copy value to input buffer (tensor)
  model_input->data.f[0] = x_val;

  // Run inference
//  TfLiteStatus invoke_status = interpreter->Invoke();
//  if (invoke_status != kTfLiteOk) {
//    error_reporter->Report("Invoke failed on input: %f\n", x_val);
//  }

  // Read predicted y value from output buffer (tensor)
  float y_val = model_output->data.f[0];

  // Translate to a PWM LED brightness
//  int brightness = (int)(255 * y_val);
//  analogWrite(led_pin, brightness);

  // Print value
//  Serial.println(y_val);

#if DEBUG
  Serial.print("Time for inference (us): ");
  Serial.println(micros() - start_timestamp);
#endif
}
