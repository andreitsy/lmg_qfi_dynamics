!> Result files and number formatting.
!>
!> The CSV layout is the one written by the Python package
!> (`time,m_x,m_y,m_z,qfi,qfi_raw`), so the files can be loaded, compared
!> and plotted with the existing tooling.
module lmg_io
   use lmg_kinds, only: wp, dp, ik
   use lmg_dynamics, only: qfi_point
   implicit none
   private

   public :: write_csv, format_real, format_time, format_fixed2, ensure_directory, result_file_name

contains

   !> Decimal representation of a 128-bit time.
   pure function format_time(t) result(s)
      integer(ik), intent(in) :: t
      character(len=:), allocatable :: s
      character(len=48) :: buf

      write (buf, '(i0)') t
      s = trim(buf)
   end function format_time

   !> Shortest general format with `digits` significant digits.
   pure function format_real(x, digits) result(s)
      real(wp), intent(in) :: x
      integer, intent(in)  :: digits
      character(len=:), allocatable :: s
      character(len=96) :: buf
      character(len=24) :: fmt

      write (fmt, '(a,i0,a)') '(g0.', digits, ')'
      write (buf, fmt) x
      s = trim(adjustl(buf))
   end function format_real

   !> Fixed notation with two decimals, like Python's f"{x:.2f}".
   pure function format_fixed2(x) result(s)
      real(wp), intent(in) :: x
      character(len=:), allocatable :: s
      character(len=64) :: buf

      write (buf, '(f40.2)') real(x, dp)
      s = trim(adjustl(buf))
   end function format_fixed2

   !> <dir>/<prefix><STATE>_N=<N>_B=<B>.csv
   pure function result_file_name(dir, prefix, state, nspins, b) result(path)
      character(len=*), intent(in) :: dir, prefix, state
      integer, intent(in)          :: nspins
      real(wp), intent(in)         :: b
      character(len=:), allocatable :: path
      character(len=16) :: nbuf

      write (nbuf, '(i0)') nspins
      path = trim(dir)//'/'//prefix//state//'_N='//trim(nbuf)//'_B='//format_fixed2(b)//'.csv'
   end function result_file_name

   !> mkdir -p
   subroutine ensure_directory(path, ierr)
      character(len=*), intent(in) :: path
      integer, intent(out)         :: ierr
      integer :: cmdstat

      ierr = 0
      call execute_command_line("mkdir -p '"//trim(path)//"'", exitstat=ierr, cmdstat=cmdstat)
      if (cmdstat /= 0) ierr = cmdstat
   end subroutine ensure_directory

   !> Write one result table. Magnetisations and the normalised QFI carry 17
   !> significant digits (enough to round-trip a double); the raw QFI is
   !> written with the full quadruple-precision significand.
   subroutine write_csv(path, points, ierr)
      character(len=*), intent(in) :: path
      type(qfi_point), intent(in)  :: points(:)
      integer, intent(out)         :: ierr
      integer :: unit, i, raw_digits

      raw_digits = precision(1.0_wp) + 1
      open (newunit=unit, file=path, status='replace', action='write', iostat=ierr)
      if (ierr /= 0) return
      write (unit, '(a)') 'time,m_x,m_y,m_z,qfi,qfi_raw'
      do i = 1, size(points)
         write (unit, '(a)') format_time(points(i)%time)//','// &
            format_real(points(i)%mx, 17)//','// &
            format_real(points(i)%my, 17)//','// &
            format_real(points(i)%mz, 17)//','// &
            format_real(points(i)%qfi, 17)//','// &
            format_real(points(i)%qfi_raw, raw_digits)
      end do
      close (unit)
   end subroutine write_csv

end module lmg_io
