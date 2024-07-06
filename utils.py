import torch as t
# p(y|x) is calculated by running the model with attention head included
# p(y) is calculated by running the model without attention head included
# if finding I(S; R1, R2) then then p(y) is calculated by just turning off both attention heads.

def mutual_info_estimation(output_probs, output_probs_ablated):
    """
    Estimate mutual information between Y and X from the marginal distrobution
    p(y|x) and p(y).

    In practice p(y|x) is going to be the output logits with a part of the model not ablated
    and p(y) is going to be the output logits with some part of the model ablated.

    I feel a little icky about this becuase in reality we are doing p(output_token = y | attention_head value = regular_value)
    and p(output_token = y | attention_head value = 0) which is not the same thing as p(output_token = y)

    whatever lets implement and think later

    we'll assume that the output_token is gotten using greedy sampling I guess
    """

    output_token = t.argmax(output_probs)

    # this is just the probs at the output token
    p_y_given_x = output_probs[output_token]
    # this is just the porbs_ablated at the output token
    p_y = output_probs_ablated[output_token]

    log_ratio = t.log(p_y_given_x / p_y)

    return t.mean(log_ratio)