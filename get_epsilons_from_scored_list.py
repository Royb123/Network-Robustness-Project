
##############
#### TEST ####
##############
import random

max_eps_tst = 0.04
min_eps_tst = 0
num_of_images = 800
number = 100


def add_mistakes_to_imgs(imgs):
    num_of_mistakes = random.randint(1, num_of_images//20)
    for mistake in range(num_of_mistakes):
        indx1 = random.randint(0, len(imgs)-1)
        indx2 = random.randint(0, len(imgs)-1)
        imgs.insert(indx1,imgs.pop(indx2))
    return imgs


def real_eps(imgs):
    return [_get_real_eps(img) for img in imgs]


def _get_real_eps(img):
    return img[0] / number


def get_eps_tst(img, lower, upper):
    eps = _get_real_eps(img)
    if lower > eps:
        return lower, 1, "ERROR"
    if upper < eps:
        return upper, 1, "ERROR"
    return eps, 1, "GOOD"


src_images = sorted([[random.random() * max_eps_tst * number] for i in range(num_of_images)])
src_images_with_mistakes = add_mistakes_to_imgs(src_images[::])

images_tst = src_images_with_mistakes[::]

# print("si - ", src_images)
# print("siwm - ", src_images_with_mistakes)
# print("it - ", images_tst)

#####################
#### END OF TEST ####
#####################

MAX_EPS = max_eps_tst
MIN_EPS = min_eps_tst
images = images_tst # Images should be from the type [{}]


def get_eps(img, lower, upper):
    return get_eps_tst(img, lower, upper)


def get_all_eps_naive_way(imgs, total_lower= MIN_EPS, total_upper= MAX_EPS):
    epsilons = []
    for img in imgs:
        eps, num_of_runs, state = get_eps(img, total_lower, total_upper)
        if state != 'GOOD':
            raise Exception('total range is wrong')
        epsilons.append(eps)

    return epsilons


def get_all_eps_when_in_order(imgs, total_lower=MIN_EPS, total_upper=MAX_EPS):
    if imgs:
        mid_indx = round(len(imgs)/2)

        mid_img = imgs[mid_indx]
        mid_img_eps, num_of_runs, state = get_eps(mid_img, total_lower, total_upper)

        lower_list = imgs[:mid_indx]
        lower_eps, lower_eps_runs = get_all_eps_when_in_order(lower_list, total_lower, mid_img_eps)

        upper_list = imgs[mid_indx+1:]
        upper_eps, upper_eps_runs = get_all_eps_when_in_order(upper_list, mid_img_eps, total_upper)

        epsilon_list = lower_eps + [mid_img_eps] + upper_eps
        total_runs = num_of_runs + lower_eps_runs + upper_eps_runs
        return epsilon_list, total_runs
    else:
        return [], 0


def get_all_eps_with_mistakes_control(imgs, lower=MIN_EPS, upper=MAX_EPS):
    if imgs:
        mid_indx = round(len(imgs)/2)

        mid_img = imgs[mid_indx]
        mid_img_eps, num_of_runs, state = get_eps(mid_img, lower, upper)
        if state != 'GOOD':
            if mid_img_eps == lower:
                mid_img_eps, num_of_runs_after_mistake, state_after_mistake = get_eps(mid_img, MIN_EPS, lower)
            elif mid_img_eps == upper:
                mid_img_eps, num_of_runs_after_mistake, state_after_mistake = get_eps(mid_img, upper, MAX_EPS)
            else:
                raise Exception("Error: get_eps")
            if state_after_mistake != 'GOOD':
                raise Exception("Error: image epsilon not in boundaries")

            num_of_runs += num_of_runs_after_mistake
            new_upper = max(upper, mid_img_eps)
            new_lower = min(lower, mid_img_eps)
        else:
            new_upper = new_lower = mid_img_eps

        lower_list = imgs[:mid_indx]
        lower_eps, lower_eps_runs = get_all_eps_with_mistakes_control(lower_list, lower, new_upper)

        upper_list = imgs[mid_indx+1:]
        upper_eps, upper_eps_runs = get_all_eps_with_mistakes_control(upper_list, new_lower, upper)

        epsilon_list = lower_eps + [mid_img_eps] + upper_eps
        total_runs = num_of_runs + lower_eps_runs + upper_eps_runs
        return epsilon_list, total_runs

    else:
        return [], 0


def get_all_eps(imgs, total_lower=MIN_EPS, total_upper=MAX_EPS):
    return get_all_eps_with_mistakes_control(imgs, total_lower, total_upper)


def test():
    real = real_eps(images)
    mine, runs = get_all_eps(images)
    print("REAL:", real)
    print("MINE:", mine)
    print(real == mine)
    print('times:', runs)
test()