cd ./DCASE2025

unzip metadata_dev.zip && rm metadata_dev.zip
unzip stereo_dev.zip && rm stereo_dev.zip
unzip stereo_eval.zip && rm stereo_eval.zip

# # 解压外层文件（支持常见命名格式）
# [ -f "files-archive" ] && unzip files-archive && rm files-archive
# # [ -f "files-archive.zip" ] && unzip files-archive.zip && rm files-archive.zip

# # 一次性处理所有嵌套ZIP文件
# find . -type f -name '*.zip' -exec sh -c \
#     'unzip -q "$1" -d "$(dirname "$1")" && rm "$1"' _ {} \;