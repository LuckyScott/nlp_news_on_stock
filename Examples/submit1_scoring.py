from scoring import *
from news_submission import *
from transform import *


abs_path   =os.path.dirname(__file__)


if __name__ == '__main__':

	
	# answer=submitfile2listofdict(os.path.join(abs_path,'answer.txt'))
	answer=submitfile2listofdict('D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T1\\answer.txt')
	# answer=submitfile2listofdict('D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T2\\answer.txt')
	# submit_test =submitfile2listofdict(os.path.join(abs_path, 'submit_test.txt'))
	# submit_test =submitfile2listofdict('D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T1\\pred_result_1.txt')
	submit_test =submitfile2listofdict('D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T1\\new_pred_result.txt')
	# submit_test =submitfile2listofdict('D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T2\\pred_result_1.txt')
	# submit_test =submitfile2listofdict('D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T2\\new_pred_result.txt')
	result=scoring(submit_test,answer)
	print(result)
	res = [result[0], result[1].tolist(), result[2].tolist(), result[3], result[4].tolist(), result[5]]
	save_dict_json(res, 'D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T1\\res_content_pred_result.txt', 'a', '\n')
	# save_dict_json(res, 'D:\\iBaguo\\work_home\\bot\\bot2017Fin-master\\data\\test\\T2\\res_content_pred_result.txt', 'a', '\n')
	print('The end!')
	