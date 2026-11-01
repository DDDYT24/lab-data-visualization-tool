"""Stage D6 into a separate private output; D2 and D3-D5 artifacts stay frozen."""
import stage_d34 as stage

stage.OUT = stage.ROOT/'outputs/desktop-d06/bundle'
stage.EXTRA_RUNTIME = ('maintenance.py','installation.py')
stage.BUILD_SCOPE = 'Private D6 candidate; native desktop acceptance and release remain open'

if __name__ == '__main__':
    stage.main()
