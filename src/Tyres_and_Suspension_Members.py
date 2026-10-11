class Tyre_Loads():

    def __init__(self):
        
        #General Vehicle Data
        self.MASS = 275.0 #kg mass of car and driver
        self.F_WEIGHT = 0.45 # % proportion of weight on front axle?
        self.VERT_COG = 0.280 #m 
        self.LONG_COG = 0.842 #m behind front axle
        self.F_TRACK = 1.200 #m? (T1) front track width?
        self.R_TRACK = 1.192 #m? T2) rear track width?
        self.WHEELBASE = 1.530 #m (L)
        self.R_TIRE_LOADED_RADIUS = 0.200 #m (L2)
        self.MU_FRONT = 1.80 
        self.MU_REAR = 1.80
        self.NUM_DRIVE_WHEELS = 2
        self.FRONT_AXEL_TO_IA_PLANE = 0.654 #m
        self.IA_TO_GROUND_PLANE = 0.253 #m
        self.VERT_COP = 0.280
        self.PERCENT_LOAD = 0.5 #%PL
        self.GRAVITY = 9.81 #m/s^2

    