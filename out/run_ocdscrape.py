import scrape_ocd_logdata as sol
import time
import multiprocessing

def process_task(task):
        driver = sol.start_driver()
        try:
            sol.scrape_data(start_page=task[0], end_page=task[1], driver=driver)
        finally:
            driver.quit()  # Close the driver after processing the batch

            time.sleep(2)  # Optional: delay between batches if necessary

def main():
    total_pages = 3372
    batch_size = 12

    tasks = [[i, i+batch_size] for i in range(1, total_pages, batch_size)]

    with multiprocessing.Pool(processes=10) as pool:
        pool.map(process_task, tasks)



if __name__ == '__main__':
    main()
