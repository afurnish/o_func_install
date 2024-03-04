program read_mesh2d_s1
    use netcdf
    implicit none
    integer :: ncid, varid, status, dimid_time, dimid_face
    integer :: num_times, num_faces
    real, allocatable :: mesh2d_s1(:,:)  ! Use REAL64 for double precision
    real :: max_val
    real, parameter :: FILL_VALUE = -999.0
    integer :: i, j
    character(len=255) :: filepath

    ! Construct the file path
    filepath = "/media/af/PN/modelling_DATA/kent_estuary_project/5.Final/" // &
               "1.friction/SCW_runs/kent_1.3.8_no_wind/kent_31_merged_map.nc"

    ! Open the NetCDF file
    status = nf90_open(filepath, NF90_NOWRITE, ncid)
    if (status /= nf90_noerr) then
        print *, "Error opening file, status: ", status
        stop
    endif

    ! Get dimension IDs for time and face
    status = nf90_inq_dimid(ncid, "time", dimid_time)
    if (status /= nf90_noerr) then
        print *, "Error getting time dimension ID, status: ", status
        stop
    endif
    status = nf90_inq_dimid(ncid, "nmesh2d_face", dimid_face)
    if (status /= nf90_noerr) then
        print *, "Error getting face dimension ID, status: ", status
        stop
    endif

    ! Get the length of dimensions
    status = nf90_inquire_dimension(ncid, dimid_time, len=num_times)
    if (status /= nf90_noerr) then
        print *, "Error inquiring time dimension, status: ", status
        stop
    endif
    status = nf90_inquire_dimension(ncid, dimid_face, len=num_faces)
    if (status /= nf90_noerr) then
        print *, "Error inquiring face dimension, status: ", status
        stop
    endif

    ! Allocate the mesh2d_s1 array based on the dimensions
    allocate(mesh2d_s1(num_times, num_faces))

    ! Get variable ID for mesh2d_s1
    status = nf90_inq_varid(ncid, "mesh2d_s1", varid)
    if (status /= nf90_noerr) then
        print *, "Error getting variable ID for mesh2d_s1, status: ", status
        stop
    endif

    ! Read the mesh2d_s1 variable
    status = nf90_get_var(ncid, varid, mesh2d_s1)
    if (status /= nf90_noerr) then
        print *, "Error reading mesh2d_s1 variable, status: ", status
        stop
    else
        print *, "mesh2d_s1 variable read successfully."
    endif

    ! Print a few values for debugging
    print *, "Some values of mesh2d_s1 (first time step):"
    do j = 1, min(num_faces, 10)  ! Print values for the first 10 faces
        if (mesh2d_s1(1, j) /= FILL_VALUE) then
            print *, "mesh2d_s1(1,", j, ") =", mesh2d_s1(1, j)
        else
            print *, "mesh2d_s1(1,", j, ") is a fill value."
        endif
    end do

    ! Assume the rest of your program logic here...

    ! Clean up
    deallocate(mesh2d_s1)
    status = nf90_close(ncid)
    if (status /= nf90_noerr) then
        print *, "Error closing file, status: ", status
    endif
end program read_mesh2d_s1
