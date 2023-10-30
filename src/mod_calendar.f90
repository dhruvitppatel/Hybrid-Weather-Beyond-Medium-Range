module mod_calendar
   !module to hold calendar stuff for hybrid and parallel reservoir 
   !calculations and this is completely independent from speedy's
   !internal calendar

   use mod_utilities, only : calendar_type

   implicit none 

   type(calendar_type) :: calendar

   contains
  
    subroutine initialize_calendar(datetime,startyear,startmonth,startday,starthour)
       type(calendar_type), intent(inout) :: datetime
       integer, intent(in)                :: startyear,startmonth,startday,starthour 

       datetime%startyear = startyear
       datetime%startmonth = startmonth
       datetime%startday = startday
       datetime%starthour = starthour
    end subroutine 

    subroutine get_current_time_delta_hour(dt_calendar,hours_elapsed)
       !Takes an initialized calendar_type object and updates the current date
       !variables
       use datetime_module, only : timedelta, datetime

       type(calendar_type), intent(inout)  :: dt_calendar
       integer, intent(in)                 :: hours_elapsed

       type(datetime) :: startdate, enddate
       type(timedelta) :: timedelta_obj

       startdate = datetime(dt_calendar%startyear,dt_calendar%startmonth,dt_calendar%startday,dt_calendar%starthour)

       timedelta_obj = timedelta(hours=hours_elapsed)

       enddate = startdate + timedelta_obj

       dt_calendar%currentyear  = enddate%getYear()
       dt_calendar%currentmonth = enddate%getMonth()
       dt_calendar%currentday   = enddate%getDay()
       dt_calendar%currenthour  = enddate%getHour()

       return
    end subroutine

    subroutine get_current_time_delta_hour_from_current(dt_calendar,hours_elapsed)
       !Takes an initialized calendar_type object and updates the current date
       !variables
       use datetime_module, only : timedelta, datetime

       type(calendar_type), intent(inout)  :: dt_calendar
       integer, intent(in)                 :: hours_elapsed

       type(datetime) :: startdate, enddate
       type(timedelta) :: timedelta_obj

       startdate = datetime(dt_calendar%currentyear,dt_calendar%currentmonth,dt_calendar%currentday,dt_calendar%currenthour)

       timedelta_obj = timedelta(hours=hours_elapsed)

       enddate = startdate + timedelta_obj

       dt_calendar%currentyear  = enddate%getYear()
       dt_calendar%currentmonth = enddate%getMonth()
       dt_calendar%currentday   = enddate%getDay()
       dt_calendar%currenthour  = enddate%getHour()

       return
    end subroutine

    subroutine leap_year_check(year,is_leap_year)
       integer, intent(in)  :: year
       logical, intent(out) :: is_leap_year

       if((mod(year,4) == 0).and.(mod(year,100) /= 0)) then
         is_leap_year = .True.
       else if(mod(year,400) == 0) then
         is_leap_year = .True.
       else
         is_leap_year = .False.
       endif
       return
    end subroutine

    subroutine numof_hours(startyear,endyear,numofhours)
      !Get the number of hours assumes you start of jan 1 of start year and end
      !dec 31 of
      !endyear
      integer, intent(in)   :: startyear, endyear
      integer, intent(out)  :: numofhours

      integer               :: years_elapsed, i 
      integer, parameter    :: hours_in_year=8760 !Number of hours in a 365 day year
      integer, parameter    :: hours_in_year_leap_year = 8784

      logical               :: is_leap_year 

      years_elapsed = endyear-startyear
      numofhours = 0 
      do i=0,years_elapsed
         call leap_year_check(startyear+i,is_leap_year) 
         if(is_leap_year) then
           numofhours = numofhours + hours_in_year_leap_year
         else 
            numofhours = numofhours + hours_in_year
         endif 
      enddo 
    end subroutine

    subroutine numof_hours_into_year(year,month,day,hour,numofhours)
      !Get the number of hours you are into the year assumes you start of jan 1 of year  
      use datetime_module, only : timedelta, datetime

      integer, intent(in)   :: year,month,day,hour
      integer, intent(out)  :: numofhours

      type (datetime) :: start_of_year, current_date
      type (timedelta) :: t
 
      current_date = datetime(year,month,day,hour)
      start_of_year = datetime(current_date%getYear(), 1, 1, 0)

      t = current_date - start_of_year

      numofhours = int(t%total_seconds()/3600) + 1
    end subroutine

    subroutine time_delta_between_two_dates(start_year,start_month,start_day,start_hour,end_year,end_month,end_day,end_hour,numofhours)
      integer, intent(in)   :: start_year,start_month,start_day,start_hour
      integer, intent(in)   :: end_year,end_month,end_day,end_hour
      integer, intent(out)  :: numofhours 

      !Local variables
      integer :: hours_into_start_year, hours_into_end_year
      integer :: hours_between_years

      if(start_year /= end_year) then 
        call numof_hours(start_year,end_year-1,hours_between_years)

        call numof_hours_into_year(start_year,start_month,start_day,start_hour,hours_into_start_year)

        call numof_hours_into_year(end_year,end_month,end_day,end_hour,hours_into_end_year)

        numofhours = hours_between_years + hours_into_end_year - hours_into_start_year

      else 
        call numof_hours_into_year(start_year,start_month,start_day,start_hour,hours_into_start_year)

        call numof_hours_into_year(end_year,end_month,end_day,end_hour,hours_into_end_year)

        numofhours = hours_into_end_year - hours_into_start_year

      endif  
     
    end subroutine 

    subroutine time_delta_between_two_dates_datetime_type(datatime1,datetime2,timedelta)
      type(calendar_type), intent(inout) :: datatime1,datetime2

      integer, intent(out)               :: timedelta

      call time_delta_between_two_dates(datatime1%currentyear,datatime1%currentmonth,datatime1%currentday,datatime1%currenthour,datetime2%currentyear,datetime2%currentmonth,datetime2%currentday,datetime2%currenthour,timedelta)
      
    end subroutine 
end module mod_calendar
