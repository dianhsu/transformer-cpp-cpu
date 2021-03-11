# transformer-cpp-cpu

-------------------------------------
用C++实现一个简单的Transformer(**进行中**)

### 参考其他模型中的参数 
| key | value | description |
| --- | --- | --- |
| DIM | `512` | Embedding的维度 |
| DEP | `20` | 句子长度 |
| DIM_HID | `2048` | FeedForwardNetwork中隐藏层宽度 |
| HEAD_SIZE | `8` | MultiHeadAttention中Head的数量 | 
| ENC_LAYER_CNT | `6` | Encoder Layer 的层数 | 
| DEC_LAYER_CNT | `6` | Decoder Layer 的层数 |

#### 最终统计：
| key | value |
| --- | --- |
| 参数量 | `176454656` |
| CPU_TIME | `127.080 s` |
|REAL_TIME | `127.243 s` |
| Memory | `711688192 bytes` |
