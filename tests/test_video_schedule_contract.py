import torch

from visual_rl.backends import hunyuan, wan
from visual_rl.sampling import shifted_schedule


def test_video_schedule_matches_native_linear_grid_not_sd3_scheduler():
    for steps in (4, 16, 30):
        nodes = torch.linspace(1, 0, steps + 1, dtype=torch.float32)
        expected = 5 * nodes / (1 + 4 * nodes)
        torch.testing.assert_close(shifted_schedule(steps, 5), expected, rtol=0, atol=0)
    torch.testing.assert_close(shifted_schedule(4, 5), torch.tensor([1., .9375, 5 / 6, .625, 0.]), rtol=0, atol=0)


def test_hunyuan_eval_bf16_recurrence_final_unrounded_endpoint():
    noise = torch.randn((1, 2, 3), dtype=torch.bfloat16, generator=torch.Generator().manual_seed(312))
    nodes = shifted_schedule(30, 5)
    seen = []
    def predict(x, t, rows):
        seen.append(x.dtype)
        return (x * .217 + t[:, None, None] * .019).to(torch.bfloat16)
    state = noise
    for i in range(30):
        field = predict(state, nodes[i].expand(1), None).float()
        expected = state.float() - nodes[i] * field
        state = (state.float() + (nodes[i + 1] - nodes[i]) * field).to(torch.bfloat16)
    actual = hunyuan.evaluation_endpoint(predict, noise, nodes)
    assert set(seen) == {torch.bfloat16}
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual, state.float())


def test_wan_eval_fp32_recurrence():
    noise = torch.randn((1, 2, 3), generator=torch.Generator().manual_seed(312))
    nodes = shifted_schedule(4, 5)
    seen = []
    def predict(x, t, rows):
        seen.append(x.dtype)
        return (x * .217 + t[:, None, None] * .019).to(torch.bfloat16)
    expected = noise
    for i in range(4):
        expected = expected + (nodes[i + 1] - nodes[i]) * predict(expected, nodes[i].expand(1), None).float()
    actual = wan.evaluation_endpoint(predict, noise, nodes)
    assert set(seen) == {torch.float32}
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
