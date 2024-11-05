LSTM CELL GRAPH

```
C_t-1 >────── x ────────── + ──────────────┬─────>C_t (size: state_size)
              │            │               │
              │            │             tanh
              │            │               │
              │      ┌──── x ────┐    ┌─── x
            [sig]   [sig]     [tanh] [sig] │
              │      │           │    │    │     
h_t-1 >─── concat ───┴───────────┴────┘    └──┬──>h_t (size: state_size)
              │                               │
             x_t (size: input_size)          fc──>out (size: output_size)

     forget gate | input gate & newC | output gate
        (sig)         (sig)    (tanh)    (sig) 
```

SMALL LSTM SIZES GRAPH

(state size: 20)          (state size: 20)
[1,1...]                  [0.1,0.2...]
         > > > LSTM > > >
[1,1...]         ^        [0.3,0.4...]
(state size: 20) ^        (state size: 20)
                 ^
           [60000, 0]
          (inp size: 2)

