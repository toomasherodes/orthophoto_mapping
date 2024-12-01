def leiaVastavKoordinaat(
    mootkava, # 1 piksel kaardil : x meetrit irl. Prolly peab eraldi funktsiooni kirjutama.
    xTeadaOlevIRL,
    yTeadaOlevIRL,
    xPildilKoordinaat,
    yPildilKoordinaat,
    xOtsitavIRL,
    yOtsitavIRL
):
    xVaheIRL = xTeadaOlevIRL - xOtsitavIRL
    yVaheIRL = yTeadaOlevIRL - yOtsitavIRL
    xPikslitLiigu = xVaheIRL / mootkava
    yPikslitLiigu = yVaheIRL / mootkava
    return (xPildilKoordinaat + xPikslitLiigu, yPildilKoordinaat + yPikslitLiigu)

## ps pole testinud veel
