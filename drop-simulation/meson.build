project(
  'drop-simulation',
  'cpp',
  version: '0.1.0',
  default_options: [
    'cpp_std=c++17',
    'warning_level=3',
    'buildtype=release',
  ],
)

opencv = dependency('opencv4', modules: ['core', 'imgproc', 'imgcodecs', 'videoio'])

executable(
  'drop_simulation',
  'src/main.cpp',
  dependencies: opencv,
  install: false,
)

