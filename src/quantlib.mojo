"""Date, term-structure, and option-pricing kernels behind mojo-quantlib."""

from std.algorithm.functional import parallelize
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import exp, log, pow, sqrt
from std.sys import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def f_at(ptr: FPtr, index: Int, scalar_mask: Int, bit: Int) -> Float64:
    return ptr[0] if scalar_mask & bit else ptr[index]


def i_at(ptr: IPtr, index: Int, scalar_mask: Int, bit: Int) -> Int64:
    return ptr[0] if scalar_mask & bit else ptr[index]


def is_leap(year: Int) -> Bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def month_length(year: Int, month: Int) -> Int:
    if month == 2:
        return 29 if is_leap(year) else 28
    if month == 4 or month == 6 or month == 9 or month == 11:
        return 30
    return 31


def date_to_serial(year: Int, month: Int, day: Int) -> Int64:
    var leaps = (year - 1) // 4 - (year - 1) // 100 + (year - 1) // 400 - 460
    var serial = 367 + 365 * (year - 1901) + leaps
    for m in range(1, month):
        serial += month_length(year, m)
    return Int64(serial + day - 1)


def serial_to_date(serial: Int, y: IPtr, m: IPtr, d: IPtr, index: Int):
    var year = 1901 + (serial - 367) // 365
    while date_to_serial(year + 1, 1, 1) <= Int64(serial):
        year += 1
    while date_to_serial(year, 1, 1) > Int64(serial):
        year -= 1
    var remaining = serial - Int(date_to_serial(year, 1, 1))
    var month = 1
    while remaining >= month_length(year, month):
        remaining -= month_length(year, month)
        month += 1
    y[index] = Int64(year)
    m[index] = Int64(month)
    d[index] = Int64(remaining + 1)


def erf_small(x: Float64) -> Float64:
    var z = x * x
    var p = 9.60497373987051638749
    p = p * z + 90.0260197203842689217
    p = p * z + 2232.00534594684319226
    p = p * z + 7003.32514112805075473
    p = p * z + 55592.3013010394962768
    var q = z + 33.5617141647503099647
    q = q * z + 521.357949780152679795
    q = q * z + 4594.32382970980127987
    q = q * z + 22629.0000613890934246
    q = q * z + 49267.3942608635921086
    return x * p / q


def erfc_positive(x: Float64) -> Float64:
    if x < 1.0:
        return 1.0 - erf_small(x)
    if x >= 8.0:
        return 0.0
    var p = 2.46196981473530512524e-10
    p = p * x + 0.564189564831068821977
    p = p * x + 7.46321056442269912687
    p = p * x + 48.6371970985681366614
    p = p * x + 196.520832956077098242
    p = p * x + 526.445194995477358631
    p = p * x + 934.528527171957607540
    p = p * x + 1027.55188689515710272
    p = p * x + 557.535335369399327526
    var q = x + 13.2281951154744992508
    q = q * x + 86.7072140885989742329
    q = q * x + 354.937778887819891062
    q = q * x + 975.708501743205489753
    q = q * x + 1823.90916687909736289
    q = q * x + 2246.33760818710981792
    q = q * x + 1656.66309194161350182
    q = q * x + 557.535340817727675546
    return exp(-x * x) * p / q


def norm_cdf(x: Float64) -> Float64:
    var scaled = x * 0.7071067811865475244
    if scaled < 0.0:
        return 0.5 * erfc_positive(-scaled)
    return 1.0 - 0.5 * erfc_positive(scaled)


def norm_pdf(x: Float64) -> Float64:
    return 0.39894228040143267794 * exp(-0.5 * x * x)


def norm_pdf_simd[width: Int](
    x: SIMD[DType.float64, width]
) -> SIMD[DType.float64, width]:
    return 0.39894228040143267794 * exp(-0.5 * x * x)


def erf_small_simd[width: Int](
    x: SIMD[DType.float64, width]
) -> SIMD[DType.float64, width]:
    var z = x * x
    var p = SIMD[DType.float64, width](9.60497373987051638749)
    p = p * z + 90.0260197203842689217
    p = p * z + 2232.00534594684319226
    p = p * z + 7003.32514112805075473
    p = p * z + 55592.3013010394962768
    var q = z + 33.5617141647503099647
    q = q * z + 521.357949780152679795
    q = q * z + 4594.32382970980127987
    q = q * z + 22629.0000613890934246
    q = q * z + 49267.3942608635921086
    return x * p / q


def erfc_positive_simd[width: Int](
    x: SIMD[DType.float64, width]
) -> SIMD[DType.float64, width]:
    var p = SIMD[DType.float64, width](2.46196981473530512524e-10)
    p = p * x + 0.564189564831068821977
    p = p * x + 7.46321056442269912687
    p = p * x + 48.6371970985681366614
    p = p * x + 196.520832956077098242
    p = p * x + 526.445194995477358631
    p = p * x + 934.528527171957607540
    p = p * x + 1027.55188689515710272
    p = p * x + 557.535335369399327526
    var q = x + 13.2281951154744992508
    q = q * x + 86.7072140885989742329
    q = q * x + 354.937778887819891062
    q = q * x + 975.708501743205489753
    q = q * x + 1823.90916687909736289
    q = q * x + 2246.33760818710981792
    q = q * x + 1656.66309194161350182
    q = q * x + 557.535340817727675546
    var rational = exp(-x * x) * p / q
    var large = x.ge(8.0).select(
        SIMD[DType.float64, width](0.0), rational
    )
    return x.lt(1.0).select(1.0 - erf_small_simd(x), large)


def norm_cdf_simd[width: Int](
    x: SIMD[DType.float64, width]
) -> SIMD[DType.float64, width]:
    var scaled = x * 0.7071067811865475244
    return scaled.lt(0.0).select(
        0.5 * erfc_positive_simd(-scaled),
        1.0 - 0.5 * erfc_positive_simd(scaled),
    )


def black_value_simd[width: Int](
    sign: SIMD[DType.float64, width],
    strike: SIMD[DType.float64, width],
    forward: SIMD[DType.float64, width],
    std_dev: SIMD[DType.float64, width],
    discount: SIMD[DType.float64, width],
    displacement: SIMD[DType.float64, width],
) -> SIMD[DType.float64, width]:
    var shifted_forward = forward + displacement
    var shifted_strike = strike + displacement
    var d1 = (
        log(shifted_forward / shifted_strike) / std_dev + 0.5 * std_dev
    )
    var d2 = d1 - std_dev
    return discount * sign * (
        shifted_forward * norm_cdf_simd(sign * d1)
        - shifted_strike * norm_cdf_simd(sign * d2)
    )


def black_value(
    option_type: Int, strike: Float64, forward: Float64, std_dev: Float64,
    discount: Float64, displacement: Float64,
) -> Float64:
    var sign = 1.0 if option_type > 0 else -1.0
    var f = forward + displacement
    var k = strike + displacement
    if std_dev <= 0.0:
        return discount * max(sign * (f - k), 0.0)
    var d1 = log(f / k) / std_dev + 0.5 * std_dev
    var d2 = d1 - std_dev
    return discount * sign * (f * norm_cdf(sign * d1) - k * norm_cdf(sign * d2))


def implied_stddev_value(
    option_type: Int, strike: Float64, forward: Float64, price: Float64,
    discount: Float64, displacement: Float64, accuracy: Float64,
    max_iterations: Int,
) -> Float64:
    var intrinsic = discount * max(
        (1.0 if option_type > 0 else -1.0) * (forward - strike), 0.0
    )
    if price <= intrinsic + accuracy:
        return 0.0
    var lo = 0.0
    var hi = 1.0
    while black_value(
        option_type, strike, forward, hi, discount, displacement
    ) < price and hi < 16.0:
        hi *= 2.0
    var shifted_forward = forward + displacement
    var shifted_strike = strike + displacement
    var x = 0.5 * (lo + hi)
    for _ in range(max_iterations):
        var value = black_value(
            option_type, strike, forward, x, discount, displacement
        )
        if value < price:
            lo = x
        else:
            hi = x
        if hi - lo <= accuracy:
            break
        var d1 = (
            log(shifted_forward / shifted_strike) / x + 0.5 * x
        )
        var vega = discount * shifted_forward * norm_pdf(d1)
        var newton = x - (value - price) / vega
        if vega > 1e-14 and newton > lo and newton < hi:
            x = newton
        else:
            x = 0.5 * (lo + hi)
    return 0.5 * (lo + hi)


def implied_stddev_simd[width: Int](
    index: Int, types: IPtr, strikes: FPtr, forwards: FPtr, prices: FPtr,
    discounts: FPtr, displacements: FPtr, result: FPtr, accuracy: Float64,
    max_iterations: Int, scalar_mask: Int,
):
    var type_values = (
        SIMD[DType.int64, width](types[0])
        if scalar_mask & 1 else types.load[width=width](index)
    )
    var type_mask = type_values.gt(0)
    var sign = type_mask.select(
        SIMD[DType.float64, width](1.0),
        SIMD[DType.float64, width](-1.0),
    )
    var strike = (
        SIMD[DType.float64, width](strikes[0])
        if scalar_mask & 2 else strikes.load[width=width](index)
    )
    var forward = (
        SIMD[DType.float64, width](forwards[0])
        if scalar_mask & 4 else forwards.load[width=width](index)
    )
    var price = (
        SIMD[DType.float64, width](prices[0])
        if scalar_mask & 8 else prices.load[width=width](index)
    )
    var discount = (
        SIMD[DType.float64, width](discounts[0])
        if scalar_mask & 16 else discounts.load[width=width](index)
    )
    var displacement = (
        SIMD[DType.float64, width](displacements[0])
        if scalar_mask & 32 else displacements.load[width=width](index)
    )
    var intrinsic = discount * max(sign * (forward - strike), 0.0)
    var lo = SIMD[DType.float64, width](0.0)
    var hi = SIMD[DType.float64, width](1.0)
    for _ in range(4):
        var needs_higher = black_value_simd(
            sign, strike, forward, hi, discount, displacement
        ).lt(price)
        if not needs_higher.reduce_or():
            break
        hi = needs_higher.select(hi * 2.0, hi)
    var shifted_forward = forward + displacement
    var shifted_strike = strike + displacement
    var x = 0.5 * (lo + hi)
    for _ in range(max_iterations):
        var value = black_value_simd(
            sign, strike, forward, x, discount, displacement
        )
        var below = value.lt(price)
        lo = below.select(x, lo)
        hi = below.select(hi, x)
        if (hi - lo).le(accuracy).reduce_and():
            break
        var d1 = log(shifted_forward / shifted_strike) / x + 0.5 * x
        var vega = discount * shifted_forward * norm_pdf_simd(d1)
        var newton = x - (value - price) / vega
        var use_newton = (
            vega.gt(1e-14) & newton.gt(lo) & newton.lt(hi)
        )
        x = use_newton.select(newton, 0.5 * (lo + hi))
    var solved = 0.5 * (lo + hi)
    result.store(
        index,
        price.le(intrinsic + accuracy).select(
            SIMD[DType.float64, width](0.0), solved
        ),
    )


def implied_stddev_gpu_kernel(
    types: IPtr, strikes: FPtr, forwards: FPtr, prices: FPtr,
    discounts: FPtr, displacements: FPtr, result: FPtr, n: Int,
    accuracy: Float64, max_iterations: Int, scalar_mask: Int,
):
    var index = Int(global_idx.x)
    if index < n:
        result[index] = implied_stddev_value(
            Int(i_at(types, index, scalar_mask, 1)),
            f_at(strikes, index, scalar_mask, 2),
            f_at(forwards, index, scalar_mask, 4),
            f_at(prices, index, scalar_mask, 8),
            f_at(discounts, index, scalar_mask, 16),
            f_at(displacements, index, scalar_mask, 32),
            accuracy, max_iterations,
        )


@export("mql_dates_to_serial")
def mql_dates_to_serial(
    years: Int, months: Int, days: Int, dst: Int, n: Int
) abi("C"):
    var yp = ip(years)
    var mp = ip(months)
    var dp = ip(days)
    var result = ip(dst)
    for i in range(n):
        result[i] = date_to_serial(Int(yp[i]), Int(mp[i]), Int(dp[i]))


@export("mql_serial_to_dates")
def mql_serial_to_dates(
    serials: Int, years: Int, months: Int, days: Int, n: Int
) abi("C"):
    var sp = ip(serials)
    var yp = ip(years)
    var mp = ip(months)
    var dp = ip(days)
    for i in range(n):
        serial_to_date(Int(sp[i]), yp, mp, dp, i)


@export("mql_year_fractions")
def mql_year_fractions(
    starts: Int, ends: Int, dst: Int, n: Int, basis: Int
) abi("C"):
    var a = ip(starts)
    var b = ip(ends)
    var result = fp(dst)
    if basis == 0 or basis == 1:
        var denom = 365.0 if basis == 0 else 360.0
        for i in range(n):
            result[i] = Float64(b[i] - a[i]) / denom
        return
    for i in range(n):
        result[i] = Float64(b[i] - a[i]) / 365.0


@export("mql_discount_curve")
def mql_discount_curve(
    node_times: Int, node_log_discounts: Int, node_count: Int,
    query_times: Int, dst: Int, query_count: Int,
) abi("C"):
    var times = fp(node_times)
    var log_discounts = fp(node_log_discounts)
    var queries = fp(query_times)
    var result = fp(dst)
    for qidx in range(query_count):
        var t = queries[qidx]
        var left = 0
        if t <= times[0]:
            left = 0
        elif t >= times[node_count - 1]:
            left = node_count - 2
        else:
            var lo = 0
            var hi = node_count - 1
            while hi - lo > 1:
                var mid = (lo + hi) // 2
                if times[mid] <= t:
                    lo = mid
                else:
                    hi = mid
            left = lo
        var weight = (t - times[left]) / (times[left + 1] - times[left])
        result[qidx] = exp(
            log_discounts[left] * (1.0 - weight)
            + log_discounts[left + 1] * weight
        )


def discount_from_rate(
    rate: Float64, t: Float64, compounding: Int, frequency: Int
) -> Float64:
    if t == 0.0:
        return 1.0
    if compounding == 0:
        return 1.0 / (1.0 + rate * t)
    if compounding == 1:
        return pow(1.0 + rate / Float64(frequency), -Float64(frequency) * t)
    return exp(-rate * t)


@export("mql_zero_curve")
def mql_zero_curve(
    node_times: Int, node_rates: Int, node_count: Int,
    query_times: Int, dst: Int, query_count: Int, compounding: Int, frequency: Int,
) abi("C"):
    var times = fp(node_times)
    var rates = fp(node_rates)
    var queries = fp(query_times)
    var result = fp(dst)
    for qidx in range(query_count):
        var t = queries[qidx]
        if t == 0.0:
            result[qidx] = 1.0
            continue
        var left = 0
        if t <= times[0]:
            left = 0
        elif t >= times[node_count - 1]:
            left = node_count - 2
        else:
            var lo = 0
            var hi = node_count - 1
            while hi - lo > 1:
                var mid = (lo + hi) // 2
                if times[mid] <= t:
                    lo = mid
                else:
                    hi = mid
            left = lo
        var weight = (t - times[left]) / (times[left + 1] - times[left])
        var rate = rates[left] * (1.0 - weight) + rates[left + 1] * weight
        result[qidx] = discount_from_rate(rate, t, compounding, frequency)


@export("mql_black_formula")
def mql_black_formula(
    option_types: Int, strikes: Int, forwards: Int, std_devs: Int,
    discounts: Int, displacements: Int, dst: Int, n: Int,
) abi("C"):
    var types = ip(option_types)
    var k = fp(strikes)
    var f = fp(forwards)
    var s = fp(std_devs)
    var disc = fp(discounts)
    var disp = fp(displacements)
    var result = fp(dst)
    for i in range(n):
        result[i] = black_value(
            Int(types[i]), k[i], f[i], s[i], disc[i], disp[i]
        )


@export("mql_black_implied_stddev")
def mql_black_implied_stddev(
    option_types: Int, strikes: Int, forwards: Int, prices: Int,
    discounts: Int, displacements: Int, dst: Int, n: Int,
    scalar_mask: Int, accuracy: Float64, max_iterations: Int,
) abi("C") -> Int:
    var types = ip(option_types)
    var k = fp(strikes)
    var f = fp(forwards)
    var prices_p = fp(prices)
    var disc = fp(discounts)
    var disp = fp(displacements)
    var result = fp(dst)
    for i in range(n):
        var option_type = i_at(types, i, scalar_mask, 1)
        var strike = f_at(k, i, scalar_mask, 2)
        var forward = f_at(f, i, scalar_mask, 4)
        var price = f_at(prices_p, i, scalar_mask, 8)
        var discount = f_at(disc, i, scalar_mask, 16)
        var displacement = f_at(disp, i, scalar_mask, 32)
        var sign = 1.0 if option_type > 0 else -1.0
        var intrinsic = discount * max(
            sign * ((forward + displacement) - (strike + displacement)), 0.0
        )
        var maximum = discount * (
            forward + displacement if option_type > 0 else strike + displacement
        )
        if price < intrinsic - accuracy or price >= maximum:
            return i + 1
    comptime parallel_threshold = 16384
    comptime chunk_size = 256
    if n < parallel_threshold:
        comptime W = simdwidthof[DType.float64]()
        var simd_end = n - n % W
        for i in range(0, simd_end, W):
            implied_stddev_simd[W](
                i, types, k, f, prices_p, disc, disp, result, accuracy,
                max_iterations, scalar_mask,
            )
        for i in range(simd_end, n):
            result[i] = implied_stddev_value(
                Int(i_at(types, i, scalar_mask, 1)),
                f_at(k, i, scalar_mask, 2),
                f_at(f, i, scalar_mask, 4),
                f_at(prices_p, i, scalar_mask, 8),
                f_at(disc, i, scalar_mask, 16),
                f_at(disp, i, scalar_mask, 32),
                accuracy, max_iterations,
            )
        return 0

    def solve_chunk(chunk: Int) {imm}:
        comptime W = simdwidthof[DType.float64]()
        var start = chunk * chunk_size
        var end = min(start + chunk_size, n)
        var simd_end = start + (end - start) // W * W
        for i in range(start, simd_end, W):
            implied_stddev_simd[W](
                i, types, k, f, prices_p, disc, disp, result, accuracy,
                max_iterations, scalar_mask,
            )
        for i in range(simd_end, end):
            result[i] = implied_stddev_value(
                Int(i_at(types, i, scalar_mask, 1)),
                f_at(k, i, scalar_mask, 2),
                f_at(f, i, scalar_mask, 4),
                f_at(prices_p, i, scalar_mask, 8),
                f_at(disc, i, scalar_mask, 16),
                f_at(disp, i, scalar_mask, 32),
                accuracy, max_iterations,
            )

    var chunks = (n + chunk_size - 1) // chunk_size
    parallelize(solve_chunk, chunks, min(chunks, 16))
    return 0


@export("mql_black_implied_stddev_gpu")
def mql_black_implied_stddev_gpu(
    option_types: Int, strikes: Int, forwards: Int, prices: Int,
    discounts: Int, displacements: Int, dst: Int, n: Int,
    scalar_mask: Int, accuracy: Float64, max_iterations: Int,
) abi("C") -> Int:
    var types = ip(option_types)
    var k = fp(strikes)
    var f = fp(forwards)
    var prices_p = fp(prices)
    var disc = fp(discounts)
    var disp = fp(displacements)
    var result = fp(dst)
    for i in range(n):
        var option_type = i_at(types, i, scalar_mask, 1)
        var strike = f_at(k, i, scalar_mask, 2)
        var forward = f_at(f, i, scalar_mask, 4)
        var price = f_at(prices_p, i, scalar_mask, 8)
        var discount = f_at(disc, i, scalar_mask, 16)
        var displacement = f_at(disp, i, scalar_mask, 32)
        var sign = 1.0 if option_type > 0 else -1.0
        var intrinsic = discount * max(
            sign * ((forward + displacement) - (strike + displacement)), 0.0
        )
        var maximum = discount * (
            forward + displacement if option_type > 0 else strike + displacement
        )
        if price < intrinsic - accuracy or price >= maximum:
            return i + 1
    try:
        var ctx = DeviceContext()
        var types_d = ctx.enqueue_create_buffer[DType.int64](
            1 if scalar_mask & 1 else n
        )
        var k_d = ctx.enqueue_create_buffer[DType.float64](
            1 if scalar_mask & 2 else n
        )
        var f_d = ctx.enqueue_create_buffer[DType.float64](
            1 if scalar_mask & 4 else n
        )
        var prices_d = ctx.enqueue_create_buffer[DType.float64](
            1 if scalar_mask & 8 else n
        )
        var disc_d = ctx.enqueue_create_buffer[DType.float64](
            1 if scalar_mask & 16 else n
        )
        var disp_d = ctx.enqueue_create_buffer[DType.float64](
            1 if scalar_mask & 32 else n
        )
        var result_d = ctx.enqueue_create_buffer[DType.float64](n)
        ctx.enqueue_copy(types_d, types)
        ctx.enqueue_copy(k_d, k)
        ctx.enqueue_copy(f_d, f)
        ctx.enqueue_copy(prices_d, prices_p)
        ctx.enqueue_copy(disc_d, disc)
        ctx.enqueue_copy(disp_d, disp)
        comptime block_size = 256
        ctx.enqueue_function[implied_stddev_gpu_kernel](
            types_d, k_d, f_d, prices_d, disc_d, disp_d, result_d, n,
            accuracy, max_iterations, scalar_mask,
            grid_dim=(n + block_size - 1) // block_size,
            block_dim=block_size,
        )
        ctx.enqueue_copy(result, result_d)
        ctx.synchronize()
        return 0
    except:
        return -2


@export("mql_black_scholes")
def mql_black_scholes(
    option_types: Int, spots: Int, strikes: Int, maturities: Int,
    rates: Int, volatilities: Int, dividends: Int,
    prices: Int, deltas: Int, gammas: Int, vegas: Int, thetas: Int, rhos: Int,
    n: Int,
) abi("C"):
    var types = ip(option_types)
    var spot = fp(spots)
    var strike = fp(strikes)
    var maturity = fp(maturities)
    var rate = fp(rates)
    var vol = fp(volatilities)
    var dividend = fp(dividends)
    var price = fp(prices)
    var delta = fp(deltas)
    var gamma = fp(gammas)
    var vega = fp(vegas)
    var theta = fp(thetas)
    var rho = fp(rhos)
    for i in range(n):
        var sign = 1.0 if types[i] > 0 else -1.0
        var t = maturity[i]
        if t <= 0.0 or vol[i] <= 0.0:
            var terminal = spot[i] - strike[i]
            price[i] = max(sign * terminal, 0.0)
            delta[i] = sign if sign * terminal > 0.0 else 0.0
            gamma[i] = 0.0
            vega[i] = 0.0
            theta[i] = 0.0
            rho[i] = 0.0
            continue
        var root_t = sqrt(t)
        var sigma_root_t = vol[i] * root_t
        var d1 = (
            log(spot[i] / strike[i])
            + (rate[i] - dividend[i] + 0.5 * vol[i] * vol[i]) * t
        ) / sigma_root_t
        var d2 = d1 - sigma_root_t
        var dq = exp(-dividend[i] * t)
        var dr = exp(-rate[i] * t)
        var nd1 = norm_cdf(sign * d1)
        var nd2 = norm_cdf(sign * d2)
        var density = norm_pdf(d1)
        price[i] = sign * (
            spot[i] * dq * nd1 - strike[i] * dr * nd2
        )
        delta[i] = sign * dq * nd1
        gamma[i] = dq * density / (spot[i] * sigma_root_t)
        vega[i] = spot[i] * dq * density * root_t
        theta[i] = (
            -spot[i] * dq * density * vol[i] / (2.0 * root_t)
            + sign * dividend[i] * spot[i] * dq * nd1
            - sign * rate[i] * strike[i] * dr * nd2
        )
        rho[i] = sign * strike[i] * t * dr * nd2


@export("mql_american_crr")
def mql_american_crr(
    option_types: Int, spots: Int, strikes: Int, maturities: Int,
    rates: Int, volatilities: Int, dividends: Int, dst: Int,
    n: Int, steps: Int, work: Int,
) abi("C"):
    var types = ip(option_types)
    var spot = fp(spots)
    var strike = fp(strikes)
    var maturity = fp(maturities)
    var rate = fp(rates)
    var vol = fp(volatilities)
    var dividend = fp(dividends)
    var result = fp(dst)
    var values = fp(work)
    for index in range(n):
        var sign = 1.0 if types[index] > 0 else -1.0
        if maturity[index] <= 0.0 or vol[index] <= 0.0:
            result[index] = max(sign * (spot[index] - strike[index]), 0.0)
            continue
        var dt = maturity[index] / Float64(steps)
        var up = exp(vol[index] * sqrt(dt))
        var down = 1.0 / up
        var growth = exp((rate[index] - dividend[index]) * dt)
        var probability = (growth - down) / (up - down)
        var step_discount = exp(-rate[index] * dt)
        var bottom = spot[index] * pow(down, Float64(steps))
        var ratio = up / down
        var node_spot = bottom
        for j in range(steps + 1):
            values[j] = max(sign * (node_spot - strike[index]), 0.0)
            node_spot *= ratio
        for reverse_step in range(steps):
            var level = steps - 1 - reverse_step
            node_spot = spot[index] * pow(down, Float64(level))
            for j in range(level + 1):
                var continuation = step_discount * (
                    probability * values[j + 1]
                    + (1.0 - probability) * values[j]
                )
                var exercise = max(sign * (node_spot - strike[index]), 0.0)
                values[j] = max(continuation, exercise)
                node_spot *= ratio
        result[index] = values[0]
