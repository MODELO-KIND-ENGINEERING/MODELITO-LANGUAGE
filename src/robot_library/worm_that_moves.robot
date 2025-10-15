robot WormBot {
    body {
        shape: box(30, 4, 4)
        stiffness: 20.0
        mass: 2.5
    }
    
    parts {
        segment1: body_part {
            position: (5, 0, 0)
            size: (4, 3, 3)
            stiffness: 40.0
            mass: 2.5
        }
        
        segment2: body_part {
            position: (10, 0, 0)
            size: (4, 3, 3)
            stiffness: 40.0
            mass: 2.5
        }
        
        segment3: body_part {
            position: (15, 0, 0)
            size: (4, 3, 3)
            stiffness: 30.0
            mass: 2.5
        }
        
        segment4: body_part {
            position: (20, 0, 0)
            size: (4, 3, 3)
            stiffness: 30.0
            mass: 2.5
        }
        
        segment5: body_part {
            position: (25, 0, 0)
            size: (4, 3, 3)
            stiffness: 15.0
            mass: 2.5
        }
    }
    
    actuator {
        gait: worm
        frequency: 0.1
        forces: {
            lift: 44.0
            push: 22.0
            swing: 16.0
        }
    }
}
