# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# cross_entropy
@handle_frontend_test(
    fn_tree="mindspore.ops.cross_entropy",
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
    ),
    dtype_and_target=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0.0,
        max_value=1.0,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    dtype_and_weights=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    reduction=st.sampled_from(["mean", "none", "sum"]),
    label_smoothing=helpers.floats(min_value=0, max_values=0.99),
)

def test_mindspore_cross_entropy(
        *,
        dtype_and_input,
        dtype_and_target,
        dtype_and_weights,
        reduction,
        label_smoothing,
        on_device,
        fn_tree,
        frontend,
        test_flags,
):
    input_dtype, input = dtype_and_input
    target_dtype, target = dtype_and_target
    weight_dtype, weights = dtype_and_weights
    helpers.test_frontend_function(
        input_dtypes=input_dtype + target_dtype + weight_dtype,
        frontend=frontend,
        test_flags=test_flags,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        target=target[0],
        weight=weights[0].reshape(-1),
        reduction=reduction,
        label_smoothing=label_smoothing,
    )