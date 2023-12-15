import os


def lines2list(pathToFile):
    with open(pathToFile, "r") as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


def get_nested_list4d(num_persons, num_sessions, num_gestures):
    data_l = list()
    for pers in range(0, num_persons):
        person = list()
        for sess in range(0, num_sessions):
            sessions = list()
            for gest in range(0, num_gestures):
                np_arrs = list()
                sessions.append(np_arrs)
            person.append(sessions)
        data_l.append(person)
    return data_l


def get_nested_list2d(num_gestures):
    ret_list = list()
    for gest in range(0, num_gestures):
        gest = list()
        ret_list.append(gest)
    return ret_list


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
