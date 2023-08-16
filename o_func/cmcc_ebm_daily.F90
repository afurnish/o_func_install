!||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
!! CMCC estuary box model places downstream of hydrology model/obs && upstream of ocean model/obs 

! Copyright by CMCC Fondation OPA Division, Giorgia Verri 
! correspondence to: giorgia.verri@cmcc.it
! Web site: https://www.estuaryboxmodel.org/
!! supporting Institutions:
! CMCC Fondation OPA Division 
! University of Bologna DIFA Department

!!Release notes:
!!   Giorgia Verri CMCC OceanLab 2017/03/07 : EBM v1 code
!!   Giorgia Verri CMCC OceanLab 2017/01/03: upgrade
!!   Saham Kurdistany and Giorgia Verri CMCC OceanLab 2019/12: dynamic Lx and Ck -> EBM v2 code
!!   Giorgia Verri CMCC OceanLab 2020/01/07: use param "side" to read the proper tidal velocity component
!!   Giorgia Verri CMCC OceanLab 2022/12/20: use param

!!Notes for the user:
!! EBM v3 Customizations to be considered
!1. paths to input files are specified and need to be updated
!2. ocean forcing from NEMO model. Note Nemo file name param and param_2 prescribed lenght
!3. tidal forcing from OTPS model: Note reading starts at column 69 or 79 following OTPS output format (lines 201 and 204)
!4. MASK on tmp_ variables: a parameter "msk" is provided as argument fo this fortran code, it depends on the ocean model output files convention: it is currently passed as thrreshold value!!

 program estuary

      use netcdf
 
! implicit none

!!! declarations I/O
    INTEGER                      :: g,i,j,t,indx,est_error,u
    CHARACTER (len=4)            :: ymd
    REAL                         :: A_surf, L_e, h, h_l, Q_m
    INTEGER                      :: nx, ny, up, down, top, nl, ne, dx, dy, lx, ly, lxy
    REAL                         :: Q_r,S_l,S_oc,Q_u,S_u,Q_l,vel_tide,Q_tide,Q_tide_cross  !input /output fields 
    REAL                         :: Fr     !River flow Froude number
    REAL                         :: Ro_s   !Lower layer ocean water density 
    !REAL                         :: Q_total !River total discharge
    !REAL                         :: Q_m  !River mean discharge
    REAL,DIMENSION (:), ALLOCATABLE         :: vel_oce_array,sal_oce_array
    REAL,DIMENSION (:), ALLOCATABLE               :: tmp_s,tmp_u,tmp_oce,depth,depth_oce,depth_vel_array
    REAL,DIMENSION (6)               :: river_param
    REAL,DIMENSION (16)            :: ocean_param
 !   LOGICAL                          :: switch_tp  !Tidal pumping switch (FALSE: off, TRUE: on)                  
 
    CHARACTER (len=100)          ::InFile_Qr,InFile_param,InFile_ocean,OutFile,InFile_vel_tide
    CHARACTER (len=10)          :: year,days,hours,jday,side
    INTEGER           :: nt_dd, nt_hh, n_year
    REAL              :: jd
    REAL,DIMENSION (:), ALLOCATABLE    ::Qr_all,level_all
    REAL,DIMENSION (:), ALLOCATABLE    ::vel_tide_all, vel_tide_array
    REAL,DIMENSION (1)                :: W_m !X Po river W_m deve essere 1 array
    REAL                  ::  mll,mul

 !! ocean model fields
    REAL          ::d_jult 
    character(60)                :: param !2018
    character(60)                :: param_2 !2018
    character(8)                 :: data
    character(4)                 :: yy
    character(2)                 :: mm,dd
    INTEGER                         :: im,id,iy,kk,id_julS,ncid,ncid_2,ids,ncstat
    integer              :: ngx,ngy,ngz,ngt
    REAL,DIMENSION (:,:), ALLOCATABLE        :: Sl_tmp,Ql_tmp,S_col_tmp,Q_l_nemo_ave_tmp
    REAL,DIMENSION (:), ALLOCATABLE        :: Sl_tmp2,Ql_tmp2,S_col_tmp2,Q_l_nemo_ave_tmp2
    REAL,DIMENSION (:), ALLOCATABLE        ::S_col,Q_l_nemo_ave,S_l_nemo_ave
    REAL,DIMENSION (:,:,:), ALLOCATABLE        :: vel_oce, sal_oce
    real, dimension(:,:,:,:), ALLOCATABLE :: sal
    real, dimension(:,:,:,:), ALLOCATABLE :: vel
    real, dimension(:), ALLOCATABLE  :: depT

    CHARACTER (len=10)       :: msk,miss
    INTEGER       :: lmsk,rmiss
    
    indx = iargc( )
      if(indx.ne.11)then
        print*,' 4 arguments must be provided: '
        print*,' 1) filename for inflowing volume flux due to river runoff'
        print*,' 2) filename for EBM parameters'
        print*,' 3) filename for OTPS barotropic velocity (zonal/merid component)'
        print*,' 4) filename for NEMO customized parameters'
        print*,' 5) river mouth location'
        print*,' 6) year'
        print*,' 7) year days'
        print*,' 8) year hours'
        print*,' 9) jday at 00'
        print*,' 10) runoff missing values'
        print*,' 11) NEMO land values on LSM'
        stop
      endif

      CALL getarg(1,InFile_Qr)
      CALL getarg(2,InFile_param)
      CALL getarg(3,InFile_vel_tide)
      CALL getarg(4,InFile_ocean)
      CALL getarg(5,side)
      CALL getarg(6,year)
      CALL getarg(7,days)
      CALL getarg(8,hours)
      CALL getarg(9,jday)
      CALL getarg(10,miss)
      CALL getarg(11,msk)

 !print*, trim(InFile_param),' ',nt_dd
     read (year,'(I10)') n_year
     read (msk,'(I10)') lmsk
     read (miss,'(I10)') rmiss
     read (days,'(I10)') nt_dd
     read (hours,'(I10)') nt_hh
     read (jday,'(F12.2)') jd
     !nt_hh = ICHAR(hours)

!!!!!!!!!!!!!!!!!!!!!!!!!! Read 1 InFile_ocean(ocean model parameters) !!!!!!!!!!!!!!!!!!!!!!!!!
    open(1, file=InFile_ocean, status='old',iostat=est_error)
        !river_info
    WRITE (*,*) 'Opening file: ', InFile_ocean
    read (1,*) ocean_param
        nx=ocean_param(1) !this is the NEMO zonal point off the outlet point 
        ny=ocean_param(2) !this is the NEMO meridionall point off the outlet point 
        up=ocean_param(3) !this is the NEMO level at top of EBM lowerlayer
        down=ocean_param(4) !this is the NEMO level at bottom of EBM lowerlayer
        top=ocean_param(5) !this is the NEMO top level (deprecated)
        nl=ocean_param(6) !num NEMO levels within EBM lowerlayer
        ne=ocean_param(7) !num NEMO levels within EBM depth 
        dx=ocean_param(8) !buffer area of NEMO ROFI: zonal points around nx
        dy=ocean_param(9) !buffer area of NEMO ROFI: meridional points around ny
        lx=ocean_param(10) !lx=2dx+1
        ly=ocean_param(11) !ly=2dy+1
        lxy=ocean_param(12) !lxy=nl*lx*ly
        ngx=ocean_param(13) !NEMO mesh points in x 
        ngy=ocean_param(14) !NEMO mesh points in y 
        ngz=ocean_param(15) !NEMO mesh points in z
        ngt=ocean_param(16) !NEMO output freq (daily files with 1 timestep)
    print*,'ngt',ngt
    print*,'lxy',lxy
!!!!!!!!!!!!!!!!!!!!!! End Read 1 !!!!!!!!!!!!!!!!

!!!!!!!!!!!!!!!!!!!!!!!!!! Read 2 InFile_param(estuary box parameters) !!!!!!!!!!!!!!!!!!!!!!!!!
    open(3, file=InFile_param, status='old',iostat=est_error)
        !river_info
    WRITE (*,*) 'Opening file: ', InFile_param
    read (3,*) river_param
    !print*, 'ebm param',size(river_param)
        A_surf=river_param(1)
        L_e=river_param(2)
        W_m=river_param(3) 
        h=river_param(4)
        h_l=river_param(5)
        Q_m=river_param(6)
    
    mul =h_l/2   !Upper layer middle
    mll =h-h_l/2 !lower layer middle
!!!!!!!!!!!!!!!!!!!!!! End Read 2 !!!!!!!!!!!!!!!!

  ALLOCATE ( S_col(nt_dd) )
  ALLOCATE ( Q_l_nemo_ave(nt_dd) )
  ALLOCATE ( S_l_nemo_ave(nt_dd) )
  ALLOCATE ( tmp_s(nl) )
  ALLOCATE ( tmp_u(nl) )
  ALLOCATE ( vel_oce(lx,ly,nl) )
  ALLOCATE ( sal_oce(lx,ly,nl) )
  ALLOCATE ( depth(nl) )
  
  ALLOCATE ( vel_oce_array(lxy) )
  ALLOCATE ( sal_oce_array(lxy) )
  ALLOCATE ( depth_vel_array(lxy) )
  ALLOCATE ( tmp_oce(ne) )
  ALLOCATE ( depth_oce(ne) )

  ALLOCATE ( sal(ngx,ngy,ngz,ngt) )
  ALLOCATE ( vel(ngx,ngy,ngz,ngt) )
  ALLOCATE ( depT(ngz) )
  ALLOCATE ( Qr_all(nt_dd) )
  ALLOCATE ( level_all(nt_dd) )
  ALLOCATE ( vel_tide_all(nt_hh) )
  ALLOCATE ( vel_tide_array(nt_dd) )
    
!!!---- Open CMCC EBM output file
  est_error = 0
 ymd=InFile_ocean(17:20) !year
 OutFile='cmcc_ebm_daily_output_pogoro_'//trim(ymd)//'.txt'
 write(*,*) OutFile 
 open(file=trim(OutFile),unit=1,access='append')

!!!!!!!!!!!!!!!!!!!!!!!!! Read 3 InFile_Qr !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    open(2, file=trim(InFile_Qr), status='old',iostat=est_error)
    WRITE (*,*) 'Opening file: ', InFile_Qr 

    read (2,*) Qr_all
    !Q_total = SUM(Qr_all(:),MASK = Qr_all(:) .NE. rmiss)
    !!print*, Q_total
    !count_dd = COUNT(MASK = Qr_all(:) .NE. rmiss)
    !!print*, count_dd
    !Q_m = Q_total/count_dd
    print*, A_surf, L_e, W_m, h, h_l,Q_m

    if (Q_m .gt. 0 .AND. Q_m .lt. 100000) then
            print *,'reasonable Qmean'
    else
            stop 'stopped: potential error in runoff file'
    end if
    where (Qr_all .EQ. rmiss) Qr_all= 0./0.


!!!!!!!!!!!!!!!!!!!!! Read 4 InFile_vel_tide (OTPS 1/30deg hourly output files) !!!!!!!!!!!!!!!!!
  open(8, file=trim(InFile_vel_tide), status='old',iostat=est_error)
   WRITE (*,*) 'Opening file: ', InFile_vel_tide 
  IF ( (side=='west') .OR. (side=='east') ) THEN
  read (8,102) vel_tide_all  !uom: cm/s
  102 FORMAT(69X,F7.3)
  ELSE
  read (8,103) vel_tide_all  !uom: cm/s
  103 FORMAT(79X,F7.3)
  ENDIF
  vel_tide_all = vel_tide_all*0.01 !uom: m/s

!! Note: tidally averaged estuarine circulation
 !thus higher freq modes need to be parameterized:
 !these modes are tidal pumping and tidal mixing due to barotropic tidal inflow

 !This implies a wetting&Drying BC on hourly tidal velocity:
 !we consider only the "inflowing" tidal velocity to compute tidal pumping & mixing
!!compute daily means of inland tidal velocity from hourly data
 j=1
  do i=1,nt_hh-23,24
   IF ( (side=='west') .OR. (side=='south') ) THEN
   vel_tide_array(j) = -SUM(vel_tide_all(i:i+23),MASK = vel_tide_all(i:i+23) .LT. 0.0)
   count_pos = COUNT(MASK = vel_tide_all(i:i+23) .LT. 0.0)  
   ELSE
   vel_tide_array(j) = SUM(vel_tide_all(i:i+23),MASK = vel_tide_all(i:i+23) .GT. 0.0)
   count_pos = COUNT(MASK = vel_tide_all(i:i+23) .GT. 0.0)  
   ENDIF
 
   vel_tide_array(j) = vel_tide_array(j)/count_pos
   j=j+1
  enddo


!!!!!!!!!!!!!!!!!!!!!!!!!! Read 5 InFile_Sl_Ql (NEMO daily output files) !!!!!!!!!!!!!!!!!!!!!!!!!    
DO kk=1,nt_dd
 
 d_jult=kk+jd; id_julS=int(d_jult)
 print*,'jday',d_jult
 
 call CALDAT(id_julS,im,id,iy);write(data,'(i2.2,i2.2,i4.4)')id,im,iy
  write(yy,'(i4.4)')iy; write(mm,'(i2.2)')im; write(dd,'(i2.2)')id
 !i2.2 :: to get zero before unit mm and dd
 ! read file with salinity
  param= yy//mm//dd//'_PSAL_NA.nc' 

  IF ( (side=='west') .OR. (side=='east') ) THEN
 ! read file with zonal component of vel
  param_2= yy//mm//dd//'_RFVL_NA.nc' 
  ELSE
 ! read file with meridional component of vel
  param_2= yy//mm//dd//'_RFVL_NA.nc'
  ENDIF 
 
 print*,'param  ',param
 print*,'param_2  ',param_2

 !NEMO data on selected year 
  ncstat=nf90_open(trim('/work/opa/gv29119/EBMv2_website/CMEMS_'//ymd//'/'//param), nf90_nowrite, ncid)
  ncstat_2=nf90_open(trim('/work/opa/gv29119/EBMv2_website/CMEMS_'//ymd//'/'//param_2), nf90_nowrite, ncid_2)
  call handle_err(ncstat)
  print*,'I am here: open'

!-------  Read NEMO fields------------------
      ncstat = nf90_inq_varid (ncid, 'so', ids) !2018
      !ncstat = nf90_inq_varid (ncid, 'vosaline', ids)
      call handle_err(ncstat)
      ncstat = nf90_get_var (ncid, ids, sal)
      call handle_err(ncstat)
      !ncstat = nf90_inq_varid (ncid, 'deptht', ids)  !till 2012
      ncstat = nf90_inq_varid (ncid, 'depth', ids)
      call handle_err(ncstat)
      ncstat = nf90_get_var (ncid, ids, depT)
      call handle_err(ncstat)
      ncstat = nf90_close (ncid)
      call handle_err(ncstat)

      IF ( (side=='west') .OR. (side=='east') ) THEN
        ncstat_2 = nf90_inq_varid (ncid_2, 'uo', ids) !2018
       ! ncstat_2 = nf90_inq_varid (ncid_2, 'vozocrtx', ids)
      ELSE
        ncstat_2 = nf90_inq_varid (ncid_2, 'vo', ids)
       ! ncstat_2 = nf90_inq_varid (ncid_2, 'vomecrty', ids)
      ENDIF
        ncstat_2 = nf90_get_var (ncid_2, ids, vel)
        ncstat_2 = nf90_close (ncid_2)


!NEMO depth at river mouth  
    depth(:)=depT(up:down) !nemo levels corresponding to estaury box lowerlayer 
    depth_oce(:)=depT(top:down) !nemo levels corresponding to estaury box 

!NEMO Salinity at river mouth
    !print*,'size sal dims',size(sal,dim=1),size(sal,dim=3) 
    tmp_s(:)=sal(nx,ny,up:down,1)
    tmp_oce(:)=sal(nx,ny,top:down,1) 

!NEMO velocity comp at river mouth
    tmp_u(:)=vel(nx,ny,up:down,1) 
   
!averages of Nemo salinity through the whole column (for computing the tidal term)
!mask is applied to avoid land points: .LT. lmsk
  S_col(kk)=(SUM(tmp_oce/abs(depth_oce-h_l),MASK = tmp_oce .LT. lmsk))/(SUM(1/abs(depth_oce-h_l),MASK = tmp_oce .LT. lmsk))

!3d averages of nemo volume flux & salinity with points surrouding the mouth ( grid points) 
!As ocean model is at mesoscale, a buffer zone in the ROFI is assumed for a realistic estimate
!mask is applied to avoid land points
  vel_oce(:,:,:)=vel(nx-dx:nx+dx,ny-dy:ny+dy,up:down,1) 
  sal_oce(:,:,:)=sal(nx-dx:nx+dx,ny-dy:ny+dy,up:down,1) 
   do i=1,lxy,nl !lxy
    depth_vel_array(i:i+nl-1)=depth(:)
   enddo
 
  g=1
   do j=1,lx
   do k=1,ly
    vel_oce_array(g:g+nl-1)=vel_oce(j,k,:)
    sal_oce_array(g:g+nl-1)=sal_oce(j,k,:)
    g=g+nl
   enddo
   enddo

IF ( (side=='west') .OR. (side=='south') ) THEN
 Q_l_nemo_ave(kk)= -((SUM (vel_oce_array/abs(depth_vel_array-mll),MASK = vel_oce_array .LT. lmsk ))/(SUM(1/abs(depth_vel_array-mll),MASK = vel_oce_array .LT. lmsk)))*(W_m(1)*h_l) 
ELSE
 Q_l_nemo_ave(kk)= ((SUM (vel_oce_array/abs(depth_vel_array-mll),MASK = vel_oce_array .LT. lmsk ))/(SUM(1/abs(depth_vel_array-mll),MASK = vel_oce_array .LT. lmsk)))*(W_m(1)*h_l) 
ENDIF

 S_l_nemo_ave(kk)=((SUM (sal_oce_array/abs(depth_vel_array-mll),MASK = sal_oce_array .LT. lmsk ))/(SUM(1/abs(depth_vel_array-mll),MASK = sal_oce_array .LT. lmsk)))

END DO


!!-- Here the Temporal Loop starts
 print*, 'loop starts' 
  
!!-- Daily Temporal Loop
 do tt=1,nt_dd
   Q_r = Qr_all(tt)
   S_l = S_l_nemo_ave(tt)
   S_oc = S_col(tt)
   Q_l = Q_l_nemo_ave(tt)
   vel_tide = vel_tide_array(tt)

!!--Wetting&drying BC on ocean water inflow 
!!-- if daily Q_l <0 (oceanwater is not entering the estuary) put Q_l=0.001
  if (Q_l .lt. 0) then
   Q_l=0.001
  endif

 print*,'Wm',W_m(1)
   Q_tide = W_m(1)*H*vel_tide  
 print*,'Qtide',Q_tide

!!!!!!------call our estuary box model core------------------!!!!!!!!   
      call cmcc_ebm(Q_l,S_l,S_oc,Q_r,Q_tide,&
                              h,W_m,L_e,Lx,vel_tide,  &
                              Q_u,S_u,Fr,Ro_s,Q_m,C_k)                  
   print*,'Sl',S_l
   print*,'Ql',Q_l
   print*,'I am the loop'
   write(1,'(F12.5,F12.5,F12.5,F8.2,F8.2,F8.2,F8.3,F8.2,F8.1,F8.1)') Q_u, S_u, Q_l,Q_r,S_l,S_oc,vel_tide,Q_tide,Lx,C_k

 end do !end temporal loop 

!end main routine
end program estuary

!||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

      SUBROUTINE cmcc_ebm(Q_l,S_l,S_oc,Q_r,Q_tide,&
                             h,W_m,L_e,Lx,vel_tide,  &
                             Q_u,S_u,Fr,Ro_s,Q_m,C_k)

!****************************************************************************************************
  ! !DETAILS: Governing equations
  ! volume conservation equation
  ! salt conservation equation
  ! Hypothesis: incompressible fluid, rigid lid
  ! NB: we parameterize 2 processes with higher freq (i.e. hourly or shorter) than the tidally averaged
  ! (i.e. lunar day) estuarine circulation:
  ! 1) tidal mixing
  ! 2) tidal pumping
  ! W&D BCs:
  ! A)only "inflowing" ocean water (daily) is considered
  ! B)only "inflowing" tides (hourly) are considered 
!*****************************************************************************************************

  implicit none
  ! input variables
  real :: &
     Q_r,  & !Upper layer river volume flux at head
     Q_l,  & !Lower layer volume flux (m**3/s, positive if entering)
     Q_tide,  & !Q_tide
     S_l,   & !Lower layer ocean salinity 
     Ro_s,   & !Lower layer ocean water density
     S_oc,   & ! ocean salinity at mouth 
     h,  &   !estuary heigth (m)
     L_e,  &   !constant estuary length (m), deprecated field
     Lx,  &   !time dep estuary length (km)
     Fr,  &   !River flow Froude number
     Fr_box, & !estuary Froude number
     Eta, & !estuary self-similar character
     W_m,  &   !estuary width (m)
     vel_tide, & !barotropic tidal zonal velocity (m/s) positive if eastward
     k_x,   &  ! horiz mixing coeff x tracers (m2/s)
     Q_m   !Mean river discharge
  ! output variables
  real :: &
    Q_u,  & !Upper layer volume flux (m**3/s, positive)
    S_u,  & !Salinity at estuary upper layer (ppt)
    ur,   & !river flow velocity
    C_k   !Box coefficient

   !print*,'Q_m',Q_m
 
!!-----------------CMCC model x estuary box----------------------
!!---------------------------------------------------------------
!CMCC EBM v1: set k_x following MacCready 2010 (ref. Banas et al 2004)
!   k_x = W_m*vel_tide*0.035
! print*,'k_x',k_x

 ur = Q_r/((h/2)*W_m)
 print*,'ur',ur
   Fr=Q_r/((h/2)*W_m*((h/2)*9.81)**0.5)
 print*,'Fr',Fr
 print*,'S_l',S_l
   Ro_s = 1000*(1+((7.7*(1E-4)) * S_l))
 print*,'Ro_s',Ro_s
   Fr_box = vel_tide/((h*9.81)**0.5)
 print*,'Fr_box',Fr_box
 print*, 'vel_tide',vel_tide
   Eta = (Fr_box)**2*(W_m/h)

!!!--C_k dimensional equation--!!!
 IF (Q_r > Q_m) THEN
   C_k =((Ro_s/1000)**20)*((vel_tide/ur)**-0.5)*exp(-2000*Eta)
 ELSE
   C_k = 200*((Ro_s/1000)**20)*(vel_tide/ur)**0.1*exp(-2000*Eta)
 END IF
 print*,'C_k',C_k
 !C_k = 100

!!!--Compute horiz. diff coeff--!!!
   k_x = W_m*vel_tide*C_k
 print*,'k_x',k_x
 
!!!--volume flux conservation equation--!!!
  Q_u=Q_r+Q_l+h*vel_tide*W_m 

 print*,'Q_u',Q_u

!!!--Lx dimensional equation--!!!
 Lx=h*0.019*(Fr**-0.673)*((Ro_s/1000)**108.92)*((Q_tide/Q_r)**0.075)*((Q_l/Q_r)**-0.0098)
 !Lx=20 !uom: km
  
!!!--salt flux conservation equation--!!!
  S_u=(S_l*Q_l+h*(S_oc)*vel_tide*W_m+k_x*h*W_m*(S_oc/(Lx*1000)) )/(Q_r+Q_l+h*vel_tide*W_m)

 print*,'S_u',S_u
 print*,'Lx',Lx

  return
 end subroutine cmcc_ebm

! additional subroutines
!-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*------
      SUBROUTINE CALDAT (JULIAN,MM,ID,IYYY)
!                                                                   
!   ROUTINE CONVERTS JULIAN DAY TO MONTH, DAY, & YEAR.               
!   THIS CODE IS LIFTED FROM THE BOOK:                                
!   W.H. PRESS ET AL., NUMERICAL RECIPES, CAMBRIDGE UNIV. PRESS, 1986.  
!   THE ONLY MODIFICATION IS THAT REAL ARITHMETIC IS DONE IN R*8.
!                                                                 
!   THE ROUTINE OUTPUTS THE MONTH, DAY, AND YEAR ON WHICH THE      
!   SPECIFIED JULIAN DAY STARTED AT NOON.                           
!                                                                     
!   TO CONVERT MODIFIED JULIAN DAY, CALL THIS ROUTINE WITH              
!     JULIAN = MJD + 2400001                              
!                                                          
      PARAMETER (IGREG=2299161)
      IF (JULIAN.GE.IGREG) THEN
         JALPHA=INT((DBLE(JULIAN-1867216)-0.25D0)/36524.25D0)
         JA=JULIAN+1+JALPHA-INT(0.25D0*DBLE(JALPHA))
      ELSE
         JA=JULIAN
      ENDIF
      JB=JA+1524
      JC=INT(6680.D0+(DBLE(JB-2439870)-122.1D0)/365.25D0)
      JD=365*JC+INT(0.25D0*DBLE(JC))
      JE=INT(DBLE(JB-JD)/30.6001D0)
      ID=JB-JD-INT(30.6001D0*DBLE(JE))
      MM=JE-1
      IF (MM.GT.12) MM=MM-12
      IYYY=JC-4715
      IF (MM.GT.2) IYYY=IYYY-1
      IF (IYYY.LE.0) IYYY=IYYY-1
      RETURN
      END


      integer function err ( outstring, iret, ec )
      character *(*) outstring
      integer ec, rc, iret

      if (iret .EQ. 0) then
        err = iret
      else
        print *,"GFIO_",outstring
        iret = ec
        err = ec
      endif
      return
      end

!-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*------
      subroutine handle_err(ncstat)
      ! Subroutine di gestione errore file netCDF

     use netcdf
      integer :: ncstat

         if (ncstat .ne. nf90_noerr) then
            print *, nf90_strerror(ncstat)
            stop 'stopped'

         end if
      end subroutine handle_err


