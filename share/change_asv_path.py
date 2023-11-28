import json, os, sys

objective_list_path = 'opensinger_test_merge.json'
with open(objective_list_path) as json_data:
	objective_list = json.load(json_data)

new_opensinger_dir = '../../opensinger_test_merged'
new_musical_score_dir = '../../mpop600_test_score'

new_objective_list = {}

for singer_dir, ref_list in objective_list.items():
	new_singer_dir = os.path.join(new_opensinger_dir, os.path.basename(singer_dir))
	# print (new_singer_dir)

	cur_singer_obj_list = {}
	for musical_score, ref_audio in objective_list[singer_dir].items():
		new_musical_score = os.path.join(new_musical_score_dir, os.path.basename(musical_score))
		new_ref_audio_path = os.path.join(new_singer_dir,os.path.basename(ref_audio))
		# print (new_musical_score, new_ref_audio_path)
		cur_singer_obj_list[new_musical_score] = new_ref_audio_path

	new_objective_list[new_singer_dir] = cur_singer_obj_list


new_objective_list_path = 'opensinger_test_merge_new.json'
with open(new_objective_list_path, 'w') as f:
	json.dump(new_objective_list, f, indent=2, ensure_ascii=False)