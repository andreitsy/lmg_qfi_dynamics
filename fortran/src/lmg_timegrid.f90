!> Stroboscopic time grid.
!>
!> The default grid reproduces `generate_time_interval` of the Python package,
!>
!>   list(range(1, 100))
!>   + [int(x) for x in np.logspace(2, 4, num_points, endpoint=False)]
!>   + [int(x) for x in np.logspace(4, max_degree, num_points, endpoint=True)]
!>
!> including the double-precision rounding of NumPy, so that both codes
!> evaluate exactly the same integer times (beyond 2**53 the times are the
!> integer values of the doubles themselves). Times are 128-bit integers.
module lmg_timegrid
   use lmg_kinds, only: wp, dp, ik, pi
   implicit none
   private

   public :: max_time_degree, python_time_grid, logspace_int, read_time_grid

contains

   !> int(log10(|4 pi / gap|)) + 1: the last decade worth simulating.
   pure function max_time_degree(gap) result(degree)
      real(wp), intent(in) :: gap
      integer :: degree

      degree = int(log10(abs(4.0_wp*pi/gap))) + 1
   end function max_time_degree

   !> Time grid of the Python reference implementation.
   subroutine python_time_grid(num_points, max_degree, times)
      integer, intent(in) :: num_points, max_degree
      integer(ik), allocatable, intent(out) :: times(:)
      integer :: i

      if (num_points < 0) error stop 'python_time_grid: num-points must not be negative'
      if (max_degree <= 1) error stop 'python_time_grid: max_degree should be greater than 1'
      if (max_degree > 38) error stop 'python_time_grid: times beyond 1e38 do not fit into 128-bit integers'

      allocate (times(99 + 2*num_points))
      do i = 1, 99
         times(i) = int(i, ik)
      end do
      call logspace_int(2.0_dp, 4.0_dp, num_points, .false., times(100:99 + num_points))
      call logspace_int(4.0_dp, real(max_degree, dp), num_points, .true., &
                        times(100 + num_points:99 + 2*num_points))
   end subroutine python_time_grid

   !> [int(x) for x in numpy.logspace(start, stop, num, endpoint)].
   !>
   !> The exponents follow numpy.linspace operation by operation in double
   !> precision; 10**y is evaluated in quadruple precision and rounded to the
   !> nearest double, i.e. it is the correctly rounded power that a good libm
   !> returns.
   subroutine logspace_int(start, stop, num, endpoint, out)
      real(dp), intent(in)     :: start, stop
      integer, intent(in)      :: num
      logical, intent(in)      :: endpoint
      integer(ik), intent(out) :: out(:)

      ! volatile: i*step and the addition of `start` must stay two separately
      ! rounded operations (NumPy does not fuse them into an FMA).
      real(dp), volatile :: prod
      real(dp) :: delta, step, y, x
      integer :: i, div

      if (num <= 0) return
      div = num
      if (endpoint) div = num - 1
      delta = stop - start
      step = 0.0_dp
      if (div > 0) step = delta/real(div, dp)

      do i = 0, num - 1
         if (div > 0) then
            if (step == 0.0_dp) then
               prod = real(i, dp)/real(div, dp)
               prod = prod*delta
            else
               prod = real(i, dp)*step
            end if
         else
            prod = real(i, dp)*delta
         end if
         y = prod + start
         if (endpoint .and. num > 1 .and. i == num - 1) y = stop
         x = real(10.0_wp**real(y, wp), dp)
         out(i + 1) = int(x, ik)
      end do
   end subroutine logspace_int

   !> Read times from a text or CSV file: the first field of every line
   !> (fields end at a comma, semicolon, blank or tab). Empty lines, lines
   !> starting with '#' and a non-numeric header line are skipped, so the
   !> result files of either solver can be used directly.
   subroutine read_time_grid(path, times, ierr, message)
      character(len=*), intent(in) :: path
      integer(ik), allocatable, intent(out) :: times(:)
      integer, intent(out) :: ierr
      character(len=*), intent(out) :: message

      character(len=4096) :: line
      integer(ik), allocatable :: buf(:), grown(:)
      integer(ik) :: tval
      integer :: unit, ios, count, last, lineno
      logical :: seen_data

      ierr = 0
      message = ''
      allocate (buf(1024))
      count = 0
      lineno = 0
      seen_data = .false.

      open (newunit=unit, file=path, status='old', action='read', iostat=ios)
      if (ios /= 0) then
         ierr = 1
         message = 'cannot open time grid file '//trim(path)
         allocate (times(0))
         return
      end if
      do
         read (unit, '(a)', iostat=ios) line
         if (ios /= 0) exit
         lineno = lineno + 1
         line = adjustl(line)
         if (len_trim(line) == 0) cycle
         if (line(1:1) == '#') cycle
         last = scan(trim(line), ',; '//achar(9)) - 1
         if (last < 0) last = len_trim(line)
         if (last == 0) cycle
         read (line(1:last), *, iostat=ios) tval
         if (ios /= 0 .or. verify(line(1:last), '+-0123456789') /= 0) then
            if (.not. seen_data) cycle      ! header line
            ierr = 2
            write (message, '(a,i0,a)') 'line ', lineno, ' of '//trim(path)//' does not start with an integer time'
            exit
         end if
         if (tval < 0_ik) then
            ierr = 3
            write (message, '(a,i0,a)') 'line ', lineno, ' of '//trim(path)//': times must not be negative'
            exit
         end if
         seen_data = .true.
         if (count == size(buf)) then
            allocate (grown(2*size(buf)))
            grown(1:count) = buf(1:count)
            call move_alloc(grown, buf)
         end if
         count = count + 1
         buf(count) = tval
      end do
      close (unit)

      allocate (times(count))
      times = buf(1:count)
      if (ierr == 0 .and. count == 0) then
         ierr = 4
         message = 'no times found in '//trim(path)
      end if
   end subroutine read_time_grid

end module lmg_timegrid
