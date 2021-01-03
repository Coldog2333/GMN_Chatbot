work_dir='/Users/jiangjunfeng/mainland/private/GMN_Chatbot/'
# combine chinese csv files into one file (2011-2019)
# 2020 is left for evaluating
echo 'making data...'
python3 make_data.py --dir ${work_dir}

echo 'begin to run the training script...'
python3 main.py --dir ${work_dir}

