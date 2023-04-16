import { Component, OnInit } from '@angular/core';
import {Router} from '@angular/router'
import { MatDialog } from '@angular/material/dialog';
import { DataServiceService } from '../app/data-service.service';
import { LandingPopupComponent } from './landing-popup/landing-popup.component';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'thesis-frontend';
  sessionId: any;

  constructor(private dialog: MatDialog,private dataService: DataServiceService, private router: Router){
  }

  ngOnInit() {
    const sessionId = localStorage.getItem('sessionId');
    if (sessionId) {
      console.log(`Session already started ${sessionId}`);
      this.router.navigateByUrl('/home');
    } else {
      this.router.navigateByUrl('/login');
    }
  }

  /*showLandingPopup(): void {
    const dialogRef = this.dialog.open(LandingPopupComponent, {
      width: '250px',
      disableClose: true // add this to prevent closing the popup with ESC or clicking outside of it
    });

    dialogRef.afterClosed().subscribe(result => {
      console.log('The landing popup was closed');
      // redirect to login page only if user clicked logout
      if (result === 'logout') {
        this.router.navigate(['/login']);
        this.showLandingPopup();
      } else {
        this.router.navigateByUrl('/home');
      }
    });
  }*/

  logout() {
    if(localStorage.getItem('sessionId')!==undefined){
      localStorage.removeItem('sessionId');
    }
    console.log('Session removed');
    // pass 'logout' as the result to dialogRef.afterClosed() to trigger the redirection to login page
    //this.dialog.closeAll();
    this.router.navigateByUrl('/login');
  }
}
