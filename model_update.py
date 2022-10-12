# Helper function

def model_update(model, dim = None, K = None, x0 = None, sigma = None, r = None, \
                 div = None, T = None, dt = None, payoff_func = None):
    '''
    Modifies the option contract parameters

    Parameters
    ----------
    model : Dictionary containing all the parameters of the stock and contract
    dim : Dimension. The default is None.
    K : Strike price. The default is None.
    x0 : Initial price of the underlying securities. The default is None.
    sigma : Volatilities. The default is None.
    r : Risk-free interest rate. The default is None.
    div : Dividend yield. The default is None.
    T : Time to maturity. The default is None.
    dt : Time per step. The default is None.
    payoff_func : Option payoff structure. The default is None.

    Returns
    -------
    model : 
    '''
    if dim:
        model['dim'] = dim 
    if K: 
        model['K'] = K
    if x0:
        model['x0'] = x0
    if sigma:
        model['sigma'] = sigma
    if r:
        model['r'] = r
    if div:
        model['div'] = div
    if T:
        model['T'] = T
    if dt:
        model['dt'] = dt
    if payoff_func:
        model['payoff.func'] = payoff_func
    return None
