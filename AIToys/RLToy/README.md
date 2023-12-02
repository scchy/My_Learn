# observe preprocessing

1. org-pic (240, 256, 3) -> (240, 256, 1) -> (48, 48, 1)
   1. transforms.Grayscale
   2. transforms.Resize
2. skipFrame
   1. skip several steps
3. FrameStack
   1. package recent `num_stack` pic -> (48, 48, `num_stack`)
   2. let Agent know that mario is moving in order to quicken learning speed. 

code:
```python
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, [["right"], ["right", "A"]])
env = FrameStack(
    ResizeObservation(
        GrayScaleObservation(SkipFrame(env, skip=4)), 
        shape=84), 
    num_stack=4)

```

# 
