def get_variable_names():
    # names of variables to use:
    # column names:
    # weight, photons
    names_w_gam=[
        "weight",
        "y1.pt","y1.eta","y1.phi","y1.e",
        "y2.pt","y2.eta","y2.phi","y2.e"
    ]

    # ee samples have info on leading and sub-leading electron
    names_lep=[
        "e1.pt","e1.eta","e1.phi","e1.e",
        "e2.pt","e2.eta","e2.phi","e2.e"
    ]


    # jets: these are variable length, but we should never have > 15 ... 
    names_jets=[
        "j1.pt","j1.eta","j1.phi","j1.e","j1.isB",
        "j2.pt","j2.eta","j2.phi","j2.e","j2.isB",
        "j3.pt","j3.eta","j3.phi","j3.e","j3.isB",
        "j4.pt","j4.eta","j4.phi","j4.e","j4.isB",
        "j5.pt","j5.eta","j5.phi","j5.e","j5.isB",
        "j6.pt","j6.eta","j6.phi","j6.e","j6.isB",
        "j7.pt","j7.eta","j7.phi","j7.e","j7.isB",
        "j8.pt","j8.eta","j8.phi","j8.e","j8.isB",
        "j9.pt","j9.eta","j9.phi","j9.e","j9.isB",
        "j10.pt","j10.eta","j10.phi","j10.e","j10.isB",
        "j11.pt","j11.eta","j11.phi","j11.e","j11.isB",
        "j12.pt","j12.eta","j12.phi","j12.e","j12.isB",
        "j13.pt","j13.eta","j13.phi","j13.e","j13.isB",
        "j14.pt","j14.eta","j14.phi","j14.e","j14.isB",
        "j15.pt","j15.eta","j15.phi","j15.e","j15.isB"
    ]
        
    return(names_w_gam+names_lep+names_jets)
